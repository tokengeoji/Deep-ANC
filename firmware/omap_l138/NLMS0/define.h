/*
 * define.h
 *
 * Secondary Path Identification with Normalized LMS (NLMS)
 * Target: TI OMAP-L138 / TMS320C6748
 * Compatible with FxNLMS0 Project
 */

#ifndef DEFINE_H_
#define DEFINE_H_

#include "Codec.h"

/* =========================================================================
 * Sampling Rate and Filter Configuration (Matches FxNLMS0)
 * ========================================================================= */
#define SAMPLING_FREQ           SAMPLING_FREQ_16000     /* 16000 Hz - Matches FxNLMS0 */
#define SEC_FILTER_TAP          500                     /* Secondary path filter length S_hat(z) */

/* =========================================================================
 * NLMS Algorithm Tuning Parameters
 * ========================================================================= */
#define SEC_MU                  0.05f       /* Normalized step size (0.02 ~ 0.08 for smooth, clean impulse response) */
#define SEC_EPSILON             1e-6f       /* Small constant to avoid division by zero */
#define SEC_LEAKAGE             1.0f        /* 1.0f = Pure NLMS without artificial damping */

/* Noise Amplitude Scaling: controls output excitation noise volume */
/* 
 * 1: Full scale maximum volume (0 dBFS, peak +/-32767)
 * 2: Strong volume (-6 dBFS, peak +/-16384)
 * 4: Moderate volume (-12 dBFS)
 */
#define NOISE_AMP_DIV           1

/* =========================================================================
 * Hardware Channel Mapping Configuration
 * ========================================================================= */
/* 
 * Error Microphone Input Channel:
 * 1: Left Channel  (rx_sample >> 16, verified mic connection)
 * 2: Right Channel (rx_sample & 0xFFFF)
 * Note: Can also be changed in real-time via 'mic_channel' in CCS Expressions window!
 */
#define MIC_INPUT_CHANNEL       2

/*
 * Speaker Output Channel Selection:
 * 0: Output to BOTH Left & Right (Default like original LMS0, guaranteed to reach speaker)
 * 1: Output to RIGHT Channel ONLY (Left channel strictly 0x0000)
 * 2: Output to LEFT Channel ONLY  (Right channel strictly 0x0000)
 * Note: Can also be changed in real-time via 'speaker_channel' in CCS Expressions window!
 */
#define SPEAKER_OUTPUT_CHANNEL  0

/* Input Source: LINE_IN or MIC (AIC31 Codec) */
#define INPUT_SOURCE            LINE_IN

/* =========================================================================
 * Function Prototypes
 * ========================================================================= */
void ClearData( void );
void TriggerCoefficientDump( void );

#endif /* DEFINE_H_ */
