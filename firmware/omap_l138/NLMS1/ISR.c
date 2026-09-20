/*
 * ISR.c
 *
 * Real-Time Primary Path Identification using Normalized LMS (NLMS)
 * Target: TI OMAP-L138 / TMS320C6748
 *
 * Paper Reference:
 *   "A Hybrid SFANC-FxNLMS Algorithm for Active Noise Control based on Deep Learning"
 *   (Zhengding Luo, Dongyuan Shi, and Woon-Seng Gan, IEEE, 2022)
 *
 * Purpose:
 *   Identifies the Primary Acoustic Path P(z) between the Reference Microphone x(n)
 *   and Error Microphone d(n). The estimated impulse response P_hat(z) is required
 *   for offline pre-training of the SFANC 15-filter bank and generating training
 *   disturbances d(n) = P(z) * x(n) for the 1D CNN classifier.
 *
 * Hardware Configuration:
 *   Line_in Left  (rx_sample >> 16)   : Reference Microphone x(n) (Noise Source)
 *   Line_in Right (rx_sample & 0xFFFF): Error Microphone d(n) (Disturbance)
 *   Line_out                          : Canceling Speaker (Muted to prevent secondary path interference)
 *
 * Feature Highlights:
 *   1. Dual-Channel DC Blocker: Removes ADC DC bias on both x(n) and d(n) (< 15 Hz).
 *   2. Noise Activity Detector: Adapts only when external primary noise is present.
 *   3. NLMS Energy-Normalized Adaptation: Fast convergence (< 2s) with division safety.
 *   4. Instant Coefficient Export: Press Push Button (SW3) or set 'dump_trigger = 1'
 *      in CCS Expressions to freeze adaptation and print C-formatted array to Console.
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

/* PRNG state for optional excitation noise */
static uint32_t prng_state = 12345;

/* =========================================================================
 * Primary Path Identification Buffers
 * ========================================================================= */

/* Estimated primary path FIR filter weights P_hat(z) */
float P_hat[PRI_FILTER_TAP] = {0.0f};

/* Reference microphone input delay line: x(n), x(n-1), ... */
float X[PRI_FILTER_TAP] = {0.0f};

/* Frozen coefficient buffer for memory viewing / export */
float P_hat_final[PRI_FILTER_TAP] = {0.0f};

/* Adaptation control flags (Accessible via CCS Expressions window) */
volatile int   is_frozen         = 0;                   /* 1: Adaptation stopped, coefficients frozen */
volatile int   dump_trigger      = 0;                   /* Set to 1 in CCS to trigger coefficient dump */
volatile int   ref_channel       = REF_MIC_CHANNEL;     /* 1: Left=Ref, Right=Err / 2: Right=Ref, Left=Err */
volatile int   spk_noise_enable  = SPEAKER_OUTPUT_ENABLE;/* 0: Muted (canceling spk), 1: Output white noise */
volatile int   speaker_channel   = 0;                   /* 0: Both L+R, 1: Right-Only, 2: Left-Only (if spk enabled) */
volatile float mu                = PRI_MU;              /* Step size (0.01 ~ 0.1) */
volatile float leakage           = PRI_LEAKAGE;         /* 1.0f = Pure NLMS, 0.9999f = slight damping */
volatile float noise_gate_thresh = NOISE_GATE_THRESH;   /* Noise gate power threshold */
volatile int   reset_filter      = 0;                   /* Set to 1 in CCS to reset filter weights to 0 */

/* Real-time audio waveform buffers for CCS Graph (Length: 256) */
int16_t ref_mic_buf[MONITOR_BUFLEN] = {0};   /* Raw Left ADC (Reference mic) */
int16_t err_mic_buf[MONITOR_BUFLEN] = {0};   /* Raw Right ADC (Error mic) */
float   ref_x_buf[MONITOR_BUFLEN]   = {0.0f};/* DC-removed Reference mic x(n) */
float   err_d_buf[MONITOR_BUFLEN]   = {0.0f};/* DC-removed Error mic d(n) */
float   filter_y_buf[MONITOR_BUFLEN]= {0.0f};/* Primary path filter output y(n) */
float   error_e_buf[MONITOR_BUFLEN] = {0.0f};/* Residual error e(n) = d - y */
static int mon_idx = 0;

/* DC Blocker filter state variables */
static float dc_x_in = 0.0f, dc_x_out = 0.0f;
static float dc_d_in = 0.0f, dc_d_out = 0.0f;

/* Real-time watch variables for CCS Expressions window */
volatile float debug_ref_x      = 0.0f;     /* Reference mic sample x(n) */
volatile float debug_err_d      = 0.0f;     /* Error mic disturbance d(n) */
volatile float debug_y          = 0.0f;     /* Estimated disturbance y(n) */
volatile float debug_e          = 0.0f;     /* Error e(n) = d - y */
volatile float debug_ref_pwr    = 0.0f;     /* Reference signal power (Activity detector) */
volatile float debug_pwr        = 0.0f;     /* Vector energy ||X||^2 */
volatile float debug_err_energy = 0.0f;     /* Smoothed MSE (drops to 0 as P_hat converges) */
volatile uint32_t xrun_cnt      = 0;        /* McASP Overrun/Underrun counter */
volatile int   need_dump_print  = 0;        /* 1: Triggers safe dump in Idle Task */


/* =========================================================================
 * Buffer Initialization
 * ========================================================================= */
void ClearData( void ) {
    int k;
    for(k = 0; k < PRI_FILTER_TAP; k++) {
        P_hat[k] = 0.0f;
        X[k]     = 0.0f;
        P_hat_final[k] = 0.0f;
    }
    for(k = 0; k < MONITOR_BUFLEN; k++) {
        ref_mic_buf[k]  = 0;
        err_mic_buf[k]  = 0;
        ref_x_buf[k]    = 0.0f;
        err_d_buf[k]    = 0.0f;
        filter_y_buf[k] = 0.0f;
        error_e_buf[k]  = 0.0f;
    }
    mon_idx = 0;
    dc_x_in = 0.0f; dc_x_out = 0.0f;
    dc_d_in = 0.0f; dc_d_out = 0.0f;
    is_frozen = 0;
    dump_trigger = 0;
    debug_ref_pwr = 0.0f;
    debug_err_energy = 0.0f;
    led_cnt = 0;
}


/* =========================================================================
 * Freeze and Dump Trigger Routine
 * ========================================================================= */
void TriggerCoefficientDump( void ) {
    int k;
    is_frozen = 1;
    for(k = 0; k < PRI_FILTER_TAP; k++) {
        P_hat_final[k] = P_hat[k];
    }
    /* Turn on LED D6 to indicate coefficients frozen */
    LED_On( LED_D6 );

    /* Request safe console dump in Idle Task context */
    need_dump_print = 1;
}


/* =========================================================================
 * SWI0: Software Interrupt (Kept for SYS/BIOS app.cfg compatibility)
 * ========================================================================= */
void ProcessSwi0( void ) {
    /* Handled safely in IdleLED() task context */
}


/* =========================================================================
 * McASP Audio ISR: Real-time NLMS Primary Path Estimation
 * ========================================================================= */
void MCASP_ISR( void ) {
    uint32_t rx_sample = mcaspRegs->RBUF14;
    int k;

    /* Check McASP Overrun / Underrun flags */
    if (mcaspRegs->RSTAT & 0x01) { xrun_cnt++; mcaspRegs->RSTAT = 0x01; }
    if (mcaspRegs->XSTAT & 0x01) { xrun_cnt++; mcaspRegs->XSTAT = 0x01; }

    /* 1. Extract raw 16-bit audio from both Line-in channels */
    Int16 ch_left  = (Int16)(rx_sample >> 16);     /* Line-in Left (Tip) */
    Int16 ch_right = (Int16)(rx_sample & 0xFFFF);  /* Line-in Right (Ring) */

    /* 
     * 2. Hardware Channel Mapping:
     * ref_channel:
     *   1: Left = Reference Mic, Right = Error Mic (Default User Hardware)
     *   2: Right = Reference Mic, Left = Error Mic (Swapped Fallback)
     */
    Int16 raw_ref = (ref_channel == 2) ? ch_right : ch_left;
    Int16 raw_err = (ref_channel == 2) ? ch_left  : ch_right;

    /* Record raw samples into graph buffers for real-time visualization */
    ref_mic_buf[mon_idx] = raw_ref;
    err_mic_buf[mon_idx] = raw_err;

    /* 
     * 3. Dual-Channel DC Blocker Filter (1st-order IIR HPF, cut-off < 15 Hz)
     * Removes ADC DC offset and baseline drift from both microphones.
     */
    float raw_x_f = (float)raw_ref / 32768.0f;
    float x_n = raw_x_f - dc_x_in + 0.995f * dc_x_out;
    dc_x_in = raw_x_f;
    dc_x_out = x_n;

    float raw_d_f = (float)raw_err / 32768.0f;
    float d_n = raw_d_f - dc_d_in + 0.995f * dc_d_out;
    dc_d_in = raw_d_f;
    dc_d_out = d_n;

    /* 
     * 4. Reference Signal Power & Noise Activity Detector
     * Prevents adaptation on pure silence/background noise when external noise is OFF.
     */
    debug_ref_pwr = 0.99f * debug_ref_pwr + 0.01f * (x_n * x_n);
    int sound_active = (noise_gate_thresh <= 0.0f) || (debug_ref_pwr >= noise_gate_thresh);

    /* 
     * 5. Line_out Canceling Speaker Control:
     * By default, Line_out is MUTED (0x0000) so the canceling speaker does not
     * produce sound that would leak into the error mic and corrupt P(z) measurement.
     * Optional: If spk_noise_enable == 1, DSP can generate white noise to drive an external
     * primary noise speaker connected to Line_out.
     */
    uint16_t out_left  = 0x0000;
    uint16_t out_right = 0x0000;

    if(spk_noise_enable && !is_frozen) {
        prng_state = prng_state * 1664525 + 1013904223;
        int16_t raw_noise = (int16_t)(prng_state >> 16);
        int16_t noise_val = raw_noise / NOISE_AMP_DIV;

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
    } else {
        /* Canceling speaker is kept completely silent */
        out_left  = 0x0000;
        out_right = 0x0000;
    }

    mcaspRegs->XBUF13 = ((uint32_t)out_left << 16) | (uint32_t)out_right;

    /* 6. Check for runtime filter reset command from CCS Expressions */
    if(reset_filter) {
        reset_filter = 0;
        for(k = 0; k < PRI_FILTER_TAP; k++) {
            P_hat[k] = 0.0f;
            X[k]     = 0.0f;
        }
    }

    /* 7. Update Reference Microphone delay line: X[k] */
    for(k = PRI_FILTER_TAP - 1; k > 0; k--) {
        X[k] = X[k - 1];
    }
    X[0] = x_n;

    /* 8. Calculate estimated primary path filter output: y(n) = P_hat^T * X */
    float y = 0.0f;
    for(k = 0; k < PRI_FILTER_TAP; k++) {
        y += P_hat[k] * X[k];
    }

    /* 9. Calculate primary path estimation error: e(n) = d(n) - y(n) */
    float e = d_n - y;

    /* 10. Normalized LMS (NLMS) Weight Update */
    float pwr = PRI_EPSILON;
    if(!is_frozen && sound_active) {
        /* 10-1. Calculate energy of reference vector X: ||X||^2 */
        for(k = 0; k < PRI_FILTER_TAP; k++) {
            pwr += X[k] * X[k];
        }

        /* 10-2. Normalized step size: mu_norm = mu / (||X||^2 + epsilon) */
        float mu_norm = mu / pwr;
        if(mu_norm > 2.0f) mu_norm = 2.0f; /* Safety clamp */
        float step = mu_norm * e;

        /* 10-3. NLMS weight adaptation */
        for(k = 0; k < PRI_FILTER_TAP; k++) {
            P_hat[k] = P_hat[k] * leakage + step * X[k];
        }
    }

    /* 11. Check if CCS Expressions window requested coefficient dump */
    if(dump_trigger) {
        dump_trigger = 0;
        TriggerCoefficientDump();
    }

    /* 12. Record active signals into waveform buffers for CCS Graph */
    ref_x_buf[mon_idx]    = x_n;
    err_d_buf[mon_idx]    = d_n;
    filter_y_buf[mon_idx] = y;
    error_e_buf[mon_idx]  = e;
    if (++mon_idx >= MONITOR_BUFLEN) {
        mon_idx = 0;
    }

    /* 13. Update real-time monitoring debug variables */
    debug_ref_x      = x_n;
    debug_err_d      = d_n;
    debug_y          = y;
    debug_e          = e;
    debug_pwr        = pwr;
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
        printf("  [NLMS1] Primary Path Filter Coefficients P_hat (Taps: %d, Fs: %d Hz)\n", 
               PRI_FILTER_TAP, (int)SAMPLING_FREQ);
        printf("  Paper: A Hybrid SFANC-FxNLMS Algorithm (Zhengding Luo et al., 2022)\n");
        printf("  Usage: 1. SFANC Control Filter Pre-training (15 Fixed Filters)\n");
        printf("         2. 1D CNN Disturbance Waveform Synthesis: d(n) = P(z) * x(n)\n");
        printf("=============================================================================\n");
        printf("/* Copy and paste the block below into your Python / MATLAB / C project: */\n\n");
        printf("float P_hat[PRI_FILTER_TAP] = {\n");

        for(k = 0; k < PRI_FILTER_TAP; k++) {
            if(k % 8 == 0) {
                printf("    ");
            }
            printf("%11.8ff", P_hat_final[k]);
            if(k < PRI_FILTER_TAP - 1) {
                printf(", ");
            }
            if((k + 1) % 8 == 0) {
                printf("\n");
            }
        }
        if(PRI_FILTER_TAP % 8 != 0) {
            printf("\n");
        }
        printf("};\n\n");
        printf("=============================================================================\n");
        printf("  Coefficients frozen in P_hat_final[]. Trigger again to re-dump.\n");
        printf("=============================================================================\n\n");

        fflush(stdout);
        System_flush();
    }

    if( led_cnt >= (SAMPLING_FREQ >> 1) ) {
        LED_Toggle( LED_D4 );
        led_cnt = 0;
    }
}
