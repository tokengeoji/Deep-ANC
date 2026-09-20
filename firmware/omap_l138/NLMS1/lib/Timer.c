/***********************************************************
* 	
*	Timer Module  
*                              
*	File Name : Timer.c 
*	Date Created :
*	Last modified :
*
************************************************************/

#include "Timer.h"

void ConfigTimer_64bit( CSL_TmrRegsOvly TimerRegs, Uint32 period_high32, Uint32 period_low32 )
{
    TimerRegs->TGCR = 0;
    TimerRegs->TCR = 0;

	TimerRegs->CNTLO = 0;
	TimerRegs->CNTHI = 0;

	CSL_FINST(TimerRegs->TGCR, TMR_TGCR_TIMMODE, 64BIT_GPT);	// 64-bit time mode
	CSL_FINST(TimerRegs->TCR, TMR_TCR_CLKSRC_LO, INTERNAL);		// 24MHz internal AUXCLK

    // timer out of reset 
    TimerRegs->TGCR = CSL_FMKT(TMR_TGCR_TIMLORS, NO_RESET) | CSL_FMKT(TMR_TGCR_TIMHIRS, NO_RESET);

    // set period registers
	TimerRegs->PRDLO = period_low32;
	TimerRegs->PRDHI = period_high32;
}


void ConfigTimer_32bit( CSL_TmrRegsOvly TimerRegs, Uint32 msec )
{
    TimerRegs->TGCR = 0;
    TimerRegs->TCR = 0;

	TimerRegs->CNTLO = 0;
	TimerRegs->CNTHI = 0;

	CSL_FINST(TimerRegs->TGCR, TMR_TGCR_TIMMODE, 32BIT_UNCHAIN);	        // 32-bit time mode
	CSL_FINST(TimerRegs->TCR, TMR_TCR_CLKSRC_LO, INTERNAL);			// 24MHz internal AUXCLK

    // timer out of reset  */
	CSL_FINST(TimerRegs->TGCR, TMR_TGCR_TIMLORS, NO_RESET);

    // set period register
	TimerRegs->PRDLO = (INTERNAL_AUXCLK/1000)*msec;
}


void StartTimer(CSL_TmrRegsOvly TimerRegs)
{
	CSL_FINST(TimerRegs->TCR, TMR_TCR_ENAMODE_LO, EN_CONT);
}


void StopTimer(CSL_TmrRegsOvly TimerRegs)
{
	CSL_FINS(TimerRegs->TCR, TMR_TCR_ENAMODE_LO, 0);
	
	TimerRegs->CNTLO = 0;
	TimerRegs->CNTHI = 0;
}			
