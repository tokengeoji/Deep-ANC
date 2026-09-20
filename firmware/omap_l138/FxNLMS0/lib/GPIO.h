/***********************************************************
* 	
*	GPIO Module Header 
*                              
*	File Name : GPIO.h
*	Date Created :
*	Last modified :
*
************************************************************/

#ifndef _GPIO_H_
#define _GPIO_H_

#include "L138_common.h"

void GPIO_SetDirection( Uint8 bank, Uint32 pin_id, Uint8 direction );
void GPIO_SetOutput( Uint8 bank, Uint32 pin_id, Uint8 output );
Uint8 GPIO_GetInput( Uint8 bank, Uint32 pin_id );
void GPIO_EnableInterrupt( Uint8 bank_bit, Uint8 enable );
void GPIO_InterruptEdge( Uint8 bank, Uint32 pin_id, Uint8 edge );
Uint8 GPIO_InterruptState( Uint8 bank, Uint32 pin_id );
void GPIO_ClearInterruptState( Uint8 bank );

typedef volatile CSL_GpioRegs         *CSL_GpioRegsOvly;

extern CSL_GpioRegsOvly 	GpioRegs;

#define ENABLE	1
#define DISABLE	0

#define RISING	1
#define FALLING	2
#define BOTH	3

#define GPIO_OUT    1
#define GPIO_IN     0

#define GPIO_HIGH   1
#define GPIO_LOW    0

/**************************************************************************\
* Bank and Pin Tokens
\**************************************************************************/
#define GP0  0
#define GP1  0
#define GP2  1
#define GP3  1
#define GP4  2
#define GP5  2
#define GP6  3
#define GP7  3
#define GP8  4

#define GP0P0  (1 <<  0)
#define GP0P1  (1 <<  1)
#define GP0P2  (1 <<  2)
#define GP0P3  (1 <<  3)
#define GP0P4  (1 <<  4)
#define GP0P5  (1 <<  5)
#define GP0P6  (1 <<  6)
#define GP0P7  (1 <<  7)
#define GP0P8  (1 <<  8)
#define GP0P9  (1 <<  9)
#define GP0P10 (1 << 10)
#define GP0P11 (1 << 11)
#define GP0P12 (1 << 12)
#define GP0P13 (1 << 13)
#define GP0P14 (1 << 14)
#define GP0P15 (1 << 15)
#define GP1P0  (1 << 16)
#define GP1P1  (1 << 17)
#define GP1P2  (1 << 18)
#define GP1P3  (1 << 19)
#define GP1P4  (1 << 20)
#define GP1P5  (1 << 21)
#define GP1P6  (1 << 22)
#define GP1P7  (1 << 23)
#define GP1P8  (1 << 24)
#define GP1P9  (1 << 25)
#define GP1P10 (1 << 26)
#define GP1P11 (1 << 27)
#define GP1P12 (1 << 28)
#define GP1P13 (1 << 29)
#define GP1P14 (1 << 30)
#define GP1P15 (1 << 31)

#define GP2P0  (1 <<  0)
#define GP2P1  (1 <<  1)
#define GP2P2  (1 <<  2)
#define GP2P3  (1 <<  3)
#define GP2P4  (1 <<  4)
#define GP2P5  (1 <<  5)
#define GP2P6  (1 <<  6)
#define GP2P7  (1 <<  7)
#define GP2P8  (1 <<  8)
#define GP2P9  (1 <<  9)
#define GP2P10 (1 << 10)
#define GP2P11 (1 << 11)
#define GP2P12 (1 << 12)
#define GP2P13 (1 << 13)
#define GP2P14 (1 << 14)
#define GP2P15 (1 << 15)
#define GP3P0  (1 << 16)
#define GP3P1  (1 << 17)
#define GP3P2  (1 << 18)
#define GP3P3  (1 << 19)
#define GP3P4  (1 << 20)
#define GP3P5  (1 << 21)
#define GP3P6  (1 << 22)
#define GP3P7  (1 << 23)
#define GP3P8  (1 << 24)
#define GP3P9  (1 << 25)
#define GP3P10 (1 << 26)
#define GP3P11 (1 << 27)
#define GP3P12 (1 << 28)
#define GP3P13 (1 << 29)
#define GP3P14 (1 << 30)
#define GP3P15 (1 << 31)

#define GP4P0  (1 <<  0)
#define GP4P1  (1 <<  1)
#define GP4P2  (1 <<  2)
#define GP4P3  (1 <<  3)
#define GP4P4  (1 <<  4)
#define GP4P5  (1 <<  5)
#define GP4P6  (1 <<  6)
#define GP4P7  (1 <<  7)
#define GP4P8  (1 <<  8)
#define GP4P9  (1 <<  9)
#define GP4P10 (1 << 10)
#define GP4P11 (1 << 11)
#define GP4P12 (1 << 12)
#define GP4P13 (1 << 13)
#define GP4P14 (1 << 14)
#define GP4P15 (1 << 15)
#define GP5P0  (1 << 16)
#define GP5P1  (1 << 17)
#define GP5P2  (1 << 18)
#define GP5P3  (1 << 19)
#define GP5P4  (1 << 20)
#define GP5P5  (1 << 21)
#define GP5P6  (1 << 22)
#define GP5P7  (1 << 23)
#define GP5P8  (1 << 24)
#define GP5P9  (1 << 25)
#define GP5P10 (1 << 26)
#define GP5P11 (1 << 27)
#define GP5P12 (1 << 28)
#define GP5P13 (1 << 29)
#define GP5P14 (1 << 30)
#define GP5P15 (1 << 31)

#define GP6P0  (1 <<  0)
#define GP6P1  (1 <<  1)
#define GP6P2  (1 <<  2)
#define GP6P3  (1 <<  3)
#define GP6P4  (1 <<  4)
#define GP6P5  (1 <<  5)
#define GP6P6  (1 <<  6)
#define GP6P7  (1 <<  7)
#define GP6P8  (1 <<  8)
#define GP6P9  (1 <<  9)
#define GP6P10 (1 << 10)
#define GP6P11 (1 << 11)
#define GP6P12 (1 << 12)
#define GP6P13 (1 << 13)
#define GP6P14 (1 << 14)
#define GP6P15 (1 << 15)
#define GP7P0  (1 << 16)
#define GP7P1  (1 << 17)
#define GP7P2  (1 << 18)
#define GP7P3  (1 << 19)
#define GP7P4  (1 << 20)
#define GP7P5  (1 << 21)
#define GP7P6  (1 << 22)
#define GP7P7  (1 << 23)
#define GP7P8  (1 << 24)
#define GP7P9  (1 << 25)
#define GP7P10 (1 << 26)
#define GP7P11 (1 << 27)
#define GP7P12 (1 << 28)
#define GP7P13 (1 << 29)
#define GP7P14 (1 << 30)
#define GP7P15 (1 << 31)

#define GP8P0  (1 <<  0)
#define GP8P1  (1 <<  1)
#define GP8P2  (1 <<  2)
#define GP8P3  (1 <<  3)
#define GP8P4  (1 <<  4)
#define GP8P5  (1 <<  5)
#define GP8P6  (1 <<  6)
#define GP8P7  (1 <<  7)
#define GP8P8  (1 <<  8)
#define GP8P9  (1 <<  9)
#define GP8P10 (1 << 10)
#define GP8P11 (1 << 11)
#define GP8P12 (1 << 12)
#define GP8P13 (1 << 13)
#define GP8P14 (1 << 14)
#define GP8P15 (1 << 15)

#endif
