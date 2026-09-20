/*
 *  ======== main.c ========
 */

#include "Codec.h"
#include "I2C.h"
#include "LED_DIPSW.h"
#include "McASP.h"
#include "Timer.h"
#include "define.h"
#include "init.h"
#include <stdio.h>
#include <ti/sysbios/BIOS.h>
#include <ti/sysbios/knl/Task.h>
#include <xdc/runtime/Error.h>
#include <xdc/runtime/System.h>
#include <xdc/std.h>

/*
 *  ======== main ========
 */
Int main() {
  void ClearData(void);

  puts("\n===============================================");
  puts("           NLMS0 -- Adaptive Filter              ");
  puts("===============================================");

  SysConfigForPinMux();
  LED_DIPSW_Init();
  I2C_Init(400);
  ConfigTimer_32bit((CSL_TmrRegsOvly)CSL_TMR_1_REGS, 500);

  ClearData();

  CodecInit(SAMPLING_FREQ, WORD_LEN_16BIT, LINE_IN);
  ConfigMcASP(MCASP_32BIT, MCASP_1SLOT, CFG, RINT, NO_XINT);

  EnableInterrupt(INT_5);

  InitMcASP(RECEIVE, CFG);
  InitMcASP(TRANSMIT, CFG);

  BIOS_start(); /* does not return */
  return (0);
}
