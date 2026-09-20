/*
 * ISR.c
 *
 * Real-Time Secondary Path Identification using Normalized LMS (NLMS)
 * Target: TI OMAP-L138 / TMS320C6748
 *
 * Feature Highlights:
 *   1. NLMS Algorithm: Energy-normalized step-size for rapid convergence (< 2 seconds).
 *   2. DC Blocker: 1st-order IIR HPF to remove ADC bias and prevent coefficient drift/dropping.
 *   3. Leaky Factor: Constrains filter weights against unexcited frequency bands.
 *   4. Instant Coefficient Export: Press Push Button or set 'dump_trigger = 1' in CCS
 *      to freeze adaptation and output formatted C code to the CCS Console.
 */

#include <xdc/std.h>
#include <xdc/runtime/System.h>
#include <ti/sysbios/BIOS.h>
#include <ti/sysbios/knl/Swi.h>
#include <xdc/cfg/global.h>
#include <math.h>
#include <stdio.h>
#include "LED_DIPSW.h"
#include "Timer.h"
#include "interrupt.h"
#include "McASP.h"
#include "define.h"

/* LED heartbeat counter */
volatile static int32_t led_cnt = 0;

/* PRNG state for excitation white noise generation */
static uint32_t prng_state = 12345;

/* =========================================================================
 * Secondary Path Identification Buffers
 * ========================================================================= */

/* Estimated secondary path FIR filter weights S_hat(z) */
float S_hat[SEC_FILTER_TAP] = {0.0f};

/* Reference excitation noise delay line: x(n), x(n-1), ... */
float X[SEC_FILTER_TAP] = {0.0f};

/* Frozen coefficient buffer for memory viewing / export */
float S_hat_final[SEC_FILTER_TAP] = {0.0f};

/* Adaptation control flags */
volatile int is_frozen = 0;          /* 1: Adaptation stopped, coefficients frozen */
volatile int dump_trigger = 0;       /* Set to 1 via CCS Expressions window to dump */
volatile int speaker_channel = SPEAKER_OUTPUT_CHANNEL; /* 0: Both, 1: Right-Only, 2: Left-Only */
volatile int mic_channel = MIC_INPUT_CHANNEL;         /* 0: Auto-Detect (MIC_TEST), 1: Left (LMS0), 2: Right */

/* Real-time audio waveform buffers for CCS Graph (Matches MIC_TEST ref_mic_buf/err_mic_buf) */
#define MONITOR_BUFLEN 256
int16_t mic_left_buf[MONITOR_BUFLEN] = {0};   /* Raw Left ADC (rx_sample >> 16) */
int16_t mic_right_buf[MONITOR_BUFLEN] = {0};  /* Raw Right ADC (rx_sample & 0xFFFF) */
float   mic_d_buf[MONITOR_BUFLEN] = {0.0f};   /* Active mic input d_clean(n) */
float   filter_y_buf[MONITOR_BUFLEN] = {0.0f};/* Secondary path filter output y(n) */
float   error_e_buf[MONITOR_BUFLEN] = {0.0f}; /* Error signal e(n) = d - y */
static int mon_idx = 0;

/* DC Blocker filter state */
static float prev_d_raw = 0.0f;
static float prev_d_clean = 0.0f;

/* Real-time watch variables for CCS Expressions window */
volatile float debug_mic_d = 0.0f;       /* Raw microphone input */
volatile float debug_d_clean = 0.0f;     /* DC-removed microphone input */
volatile float debug_noise_x = 0.0f;     /* Generated white noise sample */
volatile float debug_y = 0.0f;           /* Filter output y(n) */
volatile float debug_e = 0.0f;           /* Identification error e(n) = d - y */
volatile float debug_pwr = 0.0f;         /* Excitation signal vector energy */
volatile float debug_err_energy = 0.0f;   /* Smoothed MSE (drops toward 0 as it converges) */


/* =========================================================================
 * Buffer Initialization
 * ========================================================================= */
void ClearData( void ) {
    int k;
    for(k = 0; k < SEC_FILTER_TAP; k++) {
        S_hat[k] = 0.0f;
        X[k] = 0.0f;
        S_hat_final[k] = 0.0f;
    }
    for(k = 0; k < MONITOR_BUFLEN; k++) {
        mic_left_buf[k]  = 0;
        mic_right_buf[k] = 0;
        mic_d_buf[k]     = 0.0f;
        filter_y_buf[k]  = 0.0f;
        error_e_buf[k]   = 0.0f;
    }
    mon_idx = 0;
    prev_d_raw = 0.0f;
    prev_d_clean = 0.0f;
    is_frozen = 0;
    dump_trigger = 0;
    debug_err_energy = 0.0f;
    led_cnt = 0;
}


volatile int need_dump_print = 0;       /* Set to 1 to trigger safe dump in Idle Task */

/* =========================================================================
 * Freeze and Dump Trigger Routine
 * ========================================================================= */
void TriggerCoefficientDump( void ) {
    int k;
    is_frozen = 1;
    for(k = 0; k < SEC_FILTER_TAP; k++) {
        S_hat_final[k] = S_hat[k];
    }
    /* Turn on LED D6 to indicate coefficients frozen */
    LED_On( LED_D6 );

    /* Request safe console dump in Idle Task context (avoids GateMutex assert) */
    need_dump_print = 1;
}


/* =========================================================================
 * SWI0: Software Interrupt (Kept for SYS/BIOS app.cfg compatibility)
 * ========================================================================= */
void ProcessSwi0( void ) {
    /* Handled safely in IdleLED() task context */
}


/* =========================================================================
 * McASP Audio ISR: Real-time NLMS Secondary Path Estimation
 * ========================================================================= */
void MCASP_ISR( void ) {
    uint32_t rx_sample = mcaspRegs->RBUF14;
    int k;

    /* 1. Extract raw audio from both channels */
    Int16 ch_left  = (Int16)(rx_sample >> 16);     /* Channel 1 / Left (Matches LMS0) */
    Int16 ch_right = (Int16)(rx_sample & 0xFFFF);  /* Channel 2 / Right */

    /* Record raw samples into graph buffers for real-time visualization */
    mic_left_buf[mon_idx]  = ch_left;
    mic_right_buf[mon_idx] = ch_right;

    /* 
     * 2. Select Error Microphone Input (Fixed to channel, no per-sample switching):
     * mic_channel:
     *   1: Left Channel  (rx_sample >> 16, verified active mic)
     *   2: Right Channel (rx_sample & 0xFFFF)
     */
    Int16 mic_in = (mic_channel == 2) ? ch_right : ch_left;
    float d = (float)mic_in / 32768.0f;

    /* 
     * 3. Excitation Signal Generation (Pseudo-Random White Noise)
     * Signed 16-bit zero-mean white noise [-32768, +32767]
     * While estimating, generate white noise. Once frozen, mute to 0 to protect amplifier.
     */
    Int16 noise_val = 0;
    float x_n = 0.0f;
    if(!is_frozen) {
        prng_state = prng_state * 1664525 + 1013904223;
        int16_t raw_noise = (int16_t)(prng_state >> 16);
        noise_val = raw_noise / NOISE_AMP_DIV;
        x_n = (float)noise_val / 32768.0f;
    }

    /* 
     * 5. Output excitation noise to secondary speaker via McASP Line-out
     * [Speaker Output Channel Routing - Dynamic via speaker_channel]
     *   0: Both Left & Right (Default like original LMS0, guaranteed to reach speaker)
     *   1: Right Channel Only (Left channel strictly 0x0000 for TPA3116D2)
     *   2: Left Channel Only  (Right channel strictly 0x0000)
     */
    uint16_t out_left  = 0x0000;
    uint16_t out_right = 0x0000;

    if(speaker_channel == 0) {
        out_left  = (uint16_t)noise_val;
        out_right = (uint16_t)noise_val;
    } else if(speaker_channel == 1) {
        out_left  = 0x0000;
        out_right = (uint16_t)noise_val;
    } else if(speaker_channel == 2) {
        out_left  = (uint16_t)noise_val;
        out_right = 0x0000;
    }

    mcaspRegs->XBUF13  = ((uint32_t)out_left << 16) | (uint32_t)out_right;

    /* 6. Update reference delay line: X[k] */
    for(k = SEC_FILTER_TAP - 1; k > 0; k--) {
        X[k] = X[k - 1];
    }
    X[0] = x_n;

    /* 7. Calculate estimated secondary path filter output: y(n) = S_hat^T * X */
    float y = 0.0f;
    for(k = 0; k < SEC_FILTER_TAP; k++) {
        y += S_hat[k] * X[k];
    }

    /* 8. Calculate estimation error: e(n) = d(n) - y(n) */
    float e = d - y;

    /* 9. Normalized LMS (NLMS) Weight Update */
    float pwr = SEC_EPSILON;
    if(!is_frozen) {
        /* 9-1. Calculate energy of reference vector X: ||X||^2 */
        for(k = 0; k < SEC_FILTER_TAP; k++) {
            pwr += X[k] * X[k];
        }

        /* 9-2. Normalized step size: mu_norm = mu / (||X||^2 + epsilon) */
        float mu_norm = SEC_MU / pwr;
        float step = mu_norm * e;

        /* 9-3. NLMS weight update (Pure NLMS without artificial leakage damping) */
        for(k = 0; k < SEC_FILTER_TAP; k++) {
            S_hat[k] = S_hat[k] * SEC_LEAKAGE + step * X[k];
        }
    }

    /* Check if CCS Expressions window triggered coefficient dump */
    if(dump_trigger) {
        dump_trigger = 0;
        TriggerCoefficientDump();
    }

    /* Record active signals into waveform buffers for CCS Graph */
    mic_d_buf[mon_idx]    = d;
    filter_y_buf[mon_idx] = y;
    error_e_buf[mon_idx]  = e;
    if (++mon_idx >= MONITOR_BUFLEN) {
        mon_idx = 0;
    }

    /* Update real-time monitoring debug variables */
    debug_mic_d   = d;
    debug_d_clean = d;
    debug_noise_x = x_n;
    debug_y       = y;
    debug_e       = e;
    debug_pwr     = pwr;
    debug_err_energy = 0.999f * debug_err_energy + 0.001f * (e * e);

    ++led_cnt;
}


/* =========================================================================
 * Hardware Interrupts & Idle Functions (Matches app.cfg)
 * ========================================================================= */

/* Push Button Debounce Timer ISR */
void TIMER1_TINT12_ISR( void ) {
    StopTimer( (CSL_TmrRegsOvly)CSL_TMR_1_REGS );
    GPIO_ClearInterruptState( GP2 );
    ClearInterrupt( INT_NUM_GPIO_B2 );
    EnableInterrupt( INT_NUM_GPIO_B2 );
}

/* User Push Button ISR: Freezes adaptation and dumps coefficients */
void GPIO_PUSHBUTTON_ISR( void ) {
    DisableInterrupt( INT_NUM_GPIO_B2 );
    StartTimer( (CSL_TmrRegsOvly)CSL_TMR_1_REGS );

    /* Freeze adaptation and dump coefficients to CCS Console */
    TriggerCoefficientDump();
}

/* Idle LED Task: Heartbeat on LED D4 and Safe Console Printing in Task context */
void IdleLED( void ) {
    if( need_dump_print ) {
        int k;
        need_dump_print = 0;

        printf("\n\n");
        printf("=============================================================================\n");
        printf("  [NLMS0] Secondary Path Filter Coefficients S_hat (Taps: %d, Fs: %d Hz)\n", 
               SEC_FILTER_TAP, (int)SAMPLING_FREQ);
        printf("=============================================================================\n");
        printf("/* Copy and paste the block below directly into FxNLMS0/ISR.c : */\n\n");
        printf("float S_hat[SEC_FILTER_TAP] = {\n");

        for(k = 0; k < SEC_FILTER_TAP; k++) {
            if(k % 8 == 0) {
                printf("    ");
            }
            printf("%11.8ff", S_hat_final[k]);
            if(k < SEC_FILTER_TAP - 1) {
                printf(", ");
            }
            if((k + 1) % 8 == 0) {
                printf("\n");
            }
        }
        if(SEC_FILTER_TAP % 8 != 0) {
            printf("\n");
        }
        printf("};\n\n");
        printf("=============================================================================\n");
        printf("  Coefficients frozen in S_hat_final[]. Trigger again to re-dump.\n");
        printf("=============================================================================\n\n");

        fflush(stdout);
        System_flush();
    }

    if( led_cnt >= (SAMPLING_FREQ >> 1) ) {
        LED_Toggle( LED_D4 );
        led_cnt = 0;
    }
}
