/*
 * define.h
 *
 * Real-Time Primary Path Identification using Normalized LMS (NLMS)
 * Target: TI OMAP-L138 / TMS320C6748
 * Compatible with SFANC-FxNLMS and FxNLMS0 Projects
 *
 * Hardware Configuration:
 *   Line_in Left  (rx_sample >> 16)   : Reference Microphone x(n) (Noise Source)
 *   Line_in Right (rx_sample & 0xFFFF): Error Microphone d(n) (Disturbance)
 *   Line_out                          : Canceling Speaker (Muted by default to avoid interference)
 */

#ifndef DEFINE_H_
#define DEFINE_H_

#include "Codec.h"

/* =========================================================================
 * Sampling Rate and Filter Configuration (Matches FxNLMS0 & SFANC)
 * ========================================================================= */
#define SAMPLING_FREQ           SAMPLING_FREQ_16000     /* 16000 Hz - Matches FxNLMS0 & SFANC */
#define PRI_FILTER_TAP          500                     /* Primary path filter length P_hat(z) */

/* =========================================================================
 * NLMS Algorithm Tuning Parameters
 * ========================================================================= */
#define PRI_MU                  0.05f       /* Normalized step size (0.01 ~ 0.1 for fast, clean convergence) */
#define PRI_EPSILON             1e-6f       /* Small constant to avoid division by zero */
#define PRI_LEAKAGE             1.0f        /* 1.0f = Pure NLMS (0.9999f if leakage damping needed) */

/* Noise Gate / Activity Threshold */
#define NOISE_GATE_THRESH       0.00002f    /* Minimum ref mic power to adapt (prevents adaptation on silence) */

/* Optional excitation noise scaling (if speaker_output_enable == 1) */
#define NOISE_AMP_DIV           1           /* 1: 0 dBFS, 2: -6 dBFS, 4: -12 dBFS */

/* =========================================================================
 * Hardware Channel Mapping Configuration
 * ========================================================================= */
/* 
 * Microphone Routing:
 * 1: Left = Reference Mic x(n), Right = Error Mic d(n) (Standard user hardware setup)
 * 2: Right = Reference Mic x(n), Left = Error Mic d(n) (Inverted setup fallback)
 */
#define REF_MIC_CHANNEL         1

/*
 * Line_out Canceling Speaker Control:
 * 0: Muted (0x0000) - Default: Essential during primary path measurement to avoid corrupting error mic
 * 1: Output white noise - Only when using Line_out to drive an external primary noise speaker
 */
#define SPEAKER_OUTPUT_ENABLE   0

/* Input Source: LINE_IN (AIC31 Codec) */
#define INPUT_SOURCE            LINE_IN

/* Real-time waveform monitor buffer length for CCS Graph */
#define MONITOR_BUFLEN          256

/* =========================================================================
 * Function Prototypes
 * ========================================================================= */
void ClearData( void );
void TriggerCoefficientDump( void );

#endif /* DEFINE_H_ */
