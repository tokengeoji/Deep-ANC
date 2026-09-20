/*
 * define.h
 *
 * Real-Time Active Noise Control (FxNLMS) Configuration
 * Target: TI OMAP-L138 / TMS320C6748
 * Compatible with measured secondary path from NLMS0
 */

#ifndef DEFINE_H_
#define DEFINE_H_

#include "Codec.h"

#define SAMPLING_FREQ           SAMPLING_FREQ_16000

/* FxNLMS Filter Taps */
#define FILTER_TAP              256         /* W(z) Primary adaptive control filter length */
#define SEC_FILTER_TAP          500         /* S_hat(z) Secondary path filter length (Measured 500 taps) */
#define FB_FILTER_TAP           500         /* F_hat(z) Acoustic feedback path filter length (Canceling Speaker -> Ref Mic) */

/* FxNLMS Algorithm Parameters */
#define DEFAULT_MU              0.0005f      /* Normalized step size (Tuning range: 0.001 ~ 0.02) */
#define EPSILON                 1e-6f       /* Regularization constant to avoid division by zero */
#define LEAKAGE_FACTOR          0.9999f    /* 0.99999f: prevents long-term coefficient drift (1.0f = no leakage) */

/* Hardware Channel Mapping Configuration */
#define REF_MIC_IS_LEFT         1           /* 1: Left=Ref Mic (x), Right=Err Mic (e) */
#define INV_SPEAKER_OUT         1           /* 1: Inverted anti-noise output (-y) for acoustic cancellation */
#define SPEAKER_CH_RIGHT_ONLY   1           /* 1: Output ONLY to Right channel (Left muted to 0 for TPA3116D2 protection) */

/* Real-time CCS Graph Waveform Monitoring Buffer Length */
#define MONITOR_BUFLEN          256

void ClearData( void );

#endif /* DEFINE_H_ */
