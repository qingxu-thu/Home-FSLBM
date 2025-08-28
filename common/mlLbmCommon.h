#pragma  once
#ifndef _MLLBMCOMMON_
#define _MLLBMCOMMON_

enum   MLLATTICENODE_FLAG
{
	ML_INVALID,
	ML_EMPTY,
	ML_FLUID,
	ML_FLUID_REST,
	ML_WALL,
	ML_WALL_LEFT,
	ML_WALL_RIGHT,
	ML_WALL_FOR,
	ML_WALL_BACK,
	ML_WALL_DOWN,
	ML_WALL_UP,
	ML_SOILD,
	ML_INLET,
	ML_INLET0,
	ML_INLET1,
	ML_INLET2,
	ML_OUTLET,
	ML_SMOKE,
	ML_WALL_CORNER,
	//ML_INTERFACE
	ML_XD,
	ML_YD
};

enum MLLATTICENODE_SURFACE_FLAG : unsigned char {
	TYPE_S = 0x01, // 0b00000001 // (stationary or moving) solid boundary
	TYPE_E = 0x02, // 0b00000010 // equilibrium boundary (inflow/outflow)
	TYPE_T = 0x04, // 0b00000100 // temperature boundary
	TYPE_F = 0x08, // 0b00001000 // fluid
	TYPE_I = 0x10, // 0b00010000 // interface
	TYPE_G = 0x20, // 0b00100000 // gas
	TYPE_X = 0x40, // 0b01000000 // reserved type X
	TYPE_Y = 0x80, // 0b10000000 // reserved type Y

	TYPE_MS = 0x03, // 0b00000011 // cell next to moving solid boundary
	TYPE_BO = 0x03, // 0b00000011 // any flag bit used for boundaries (temperature excluded)
	TYPE_IF = 0x18, // 0b00011000 // change from interface to fluid
	TYPE_IG = 0x30,// 0b00110000 // change from interface to gas
	TYPE_GI = 0x38,// 0b00111000 // change from gas to interface
	TYPE_SU = 0x38,// 0b00111000 // any flag bit used for SURFACE
};



#endif // !_MLLBMCOMMON_