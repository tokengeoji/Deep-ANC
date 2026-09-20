/***********************************************************
* 	
*	Timer Module Header 
*                              
*	File Name : Timer.h 
*	Date Created :
*	Last modified :
*
************************************************************/

#ifndef _TIMER_H_
#define _TIMER_H_

#include "L138_common.h"

void ConfigTimer_64bit( CSL_TmrRegsOvly TimerRegs, Uint32 period_high32, Uint32 period_low32 );
void ConfigTimer_32bit( CSL_TmrRegsOvly TimerRegs, Uint32 msec );
void StartTimer(CSL_TmrRegsOvly TimerRegs);
void StopTimer(CSL_TmrRegsOvly TimerRegs);

#define INTERNAL_AUXCLK	24000000U

/* TCR */

/*----CAPEVTMODE34 Tokens----*/
#define CSL_TMR_TCR_CAPEVTMODE_HI_RISE    (0x00000000u)
#define CSL_TMR_TCR_CAPEVTMODE_HI_FALL    (0x00000001u)
#define CSL_TMR_TCR_CAPEVTMODE_HI_BOTH    (0x00000002u)

/*----CAPMODE34 Tokens----*/
#define CSL_TMR_TCR_CAPMODE_HI_DISABLE    (0x00000000u)
#define CSL_TMR_TCR_CAPMODE_HI_ENABLE     (0x00000001u)

/*----READRSTMODE34 Tokens----*/
#define CSL_TMR_TCR_READRSTMODE_HI_CONTINUE (0x00000000u)
#define CSL_TMR_TCR_READRSTMODE_HI_RESET  (0x00000001u)

/*----TIEN34 Tokens----*/
#define CSL_TMR_TCR_TIEN_HI_NOTGATED      (0x00000000u)
#define CSL_TMR_TCR_TIEN_HI_GATED         (0x00000001u)

/*----CLKSRC34 Tokens----*/
#define CSL_TMR_TCR_CLKSRC_HI_INTERNAL    (0x00000000u)
#define CSL_TMR_TCR_CLKSRC_HI_TIMER       (0x00000001u)

/*----ENAMODE34 Tokens----*/
#define CSL_TMR_TCR_ENAMODE_HI_EN_ONCE    (0x00000001u)
#define CSL_TMR_TCR_ENAMODE_HI_EN_CONT    (0x00000002u)
#define CSL_TMR_TCR_ENAMODE_HI_EN_CONTRELOAD (0x00000003u)

/*----PWID34 Tokens----*/
#define CSL_TMR_TCR_PWID_HI_ONE_CLK       (0x00000000u)
#define CSL_TMR_TCR_PWID_HI_TWO_CLK       (0x00000001u)
#define CSL_TMR_TCR_PWID_HI_THREE_CLK     (0x00000002u)
#define CSL_TMR_TCR_PWID_HI_FOUR_CLK      (0x00000003u)

/*----CP34 Tokens----*/
#define CSL_TMR_TCR_CP_HI_PULSE           (0x00000000u)
#define CSL_TMR_TCR_CP_HI_CLOCK           (0x00000001u)

/*----INVINP34 Tokens----*/
#define CSL_TMR_TCR_INVINP_HI_NON_INVERTED (0x00000000u)
#define CSL_TMR_TCR_INVINP_HI_INVERTED    (0x00000001u)

/*----INVOUTP34 Tokens----*/
#define CSL_TMR_TCR_INVOUTP_HI_NON_INVERTED (0x00000000u)
#define CSL_TMR_TCR_INVOUTP_HI_INVERTED   (0x00000001u)

/*----TSTAT34 Tokens----*/
#define CSL_TMR_TCR_TSTAT_HI_LOW          (0x00000000u)
#define CSL_TMR_TCR_TSTAT_HI_HIGH         (0x00000001u)

/*----CAPEVTMODE12 Tokens----*/
#define CSL_TMR_TCR_CAPEVTMODE_LO_RISE    (0x00000000u)
#define CSL_TMR_TCR_CAPEVTMODE_LO_FALL    (0x00000001u)
#define CSL_TMR_TCR_CAPEVTMODE_LO_BOTH    (0x00000002u)

/*----CAPMODE12 Tokens----*/
#define CSL_TMR_TCR_CAPMODE_LO_DISABLE    (0x00000000u)
#define CSL_TMR_TCR_CAPMODE_LO_ENABLE     (0x00000001u)

/*----READRSTMODE12 Tokens----*/
#define CSL_TMR_TCR_READRSTMODE_LO_CONTINUE (0x00000000u)
#define CSL_TMR_TCR_READRSTMODE_LO_RESET  (0x00000001u)

/*----TIEN12 Tokens----*/
#define CSL_TMR_TCR_TIEN_LO_NOTGATED      (0x00000000u)
#define CSL_TMR_TCR_TIEN_LO_GATED         (0x00000001u)

/*----CLKSRC12 Tokens----*/
#define CSL_TMR_TCR_CLKSRC_LO_INTERNAL    (0x00000000u)
#define CSL_TMR_TCR_CLKSRC_LO_TIMER       (0x00000001u)

/*----ENAMODE12 Tokens----*/
#define CSL_TMR_TCR_ENAMODE_LO_EN_ONCE    (0x00000001u)
#define CSL_TMR_TCR_ENAMODE_LO_EN_CONT    (0x00000002u)
#define CSL_TMR_TCR_ENAMODE_LO_EN_CONTRELOAD (0x00000003u)

/*----PWID12 Tokens----*/
#define CSL_TMR_TCR_PWID_LO_ONE_CLK       (0x00000000u)
#define CSL_TMR_TCR_PWID_LO_TWO_CLK       (0x00000001u)
#define CSL_TMR_TCR_PWID_LO_THREE_CLK     (0x00000002u)
#define CSL_TMR_TCR_PWID_LO_FOUR_CLK      (0x00000003u)

/*----CP12 Tokens----*/
#define CSL_TMR_TCR_CP_LO_PULSE           (0x00000000u)
#define CSL_TMR_TCR_CP_LO_CLOCK           (0x00000001u)

/*----INVINP12 Tokens----*/
#define CSL_TMR_TCR_INVINP_LO_NON_INVERTED (0x00000000u)
#define CSL_TMR_TCR_INVINP_LO_INVERTED    (0x00000001u)

/*----INVOUTP12 Tokens----*/
#define CSL_TMR_TCR_INVOUTP_LO_NON_INVERTED (0x00000000u)
#define CSL_TMR_TCR_INVOUTP_LO_INVERTED   (0x00000001u)

/*----TSTAT12 Tokens----*/
#define CSL_TMR_TCR_TSTAT_LO_LOW          (0x00000000u)
#define CSL_TMR_TCR_TSTAT_LO_HIGH         (0x00000001u)


/* TGCR */

/*----PLUSEN Tokens----*/
#define CSL_TMR_TGCR_PLUSEN_DISABLE      (0x00000000u)
#define CSL_TMR_TGCR_PLUSEN_ENABLE       (0x00000001u)

/*----TIMMODE Tokens----*/
#define CSL_TMR_TGCR_TIMMODE_64BIT_GPT   (0x00000000u)
#define CSL_TMR_TGCR_TIMMODE_32BIT_UNCHAIN (0x00000001u)
#define CSL_TMR_TGCR_TIMMODE_64BIT_WDT   (0x00000002u)
#define CSL_TMR_TGCR_TIMMODE_32_CHAIN    (0x00000003u)

/*----TIM34RS Tokens----*/
#define CSL_TMR_TGCR_TIMHIRS_RESET       (0x00000000u)
#define CSL_TMR_TGCR_TIMHIRS_NO_RESET    (0x00000001u)

/*----TIM12RS Tokens----*/
#define CSL_TMR_TGCR_TIMLORS_RESET       (0x00000000u)
#define CSL_TMR_TGCR_TIMLORS_NO_RESET    (0x00000001u)

#endif
