#include "colormaps.h"
#include "colorramp.h"

namespace Mfree
{
	// Constructor
	ColorRamp::ColorRamp()
		//-----------------------------------------------------------------------------
	{
		int i;

		for (i = 0; i < 256; i++)
			rainbow((float)i / 255.0f, color_rainbow + 3 * i);

		for (i = 0; i < 256; i++)
		{
			color_grayscale[3 * i] = (float)i / 255.0f;
			color_grayscale[3 * i + 1] = (float)i / 255.0f;
			color_grayscale[3 * i + 2] = (float)i / 255.0f;
		}

		for (i = 1; i < COLOR_NUMBER; i++)
		{
			switch (i)
			{
	 
			case COLOR__MAGMA: colormaps[i] = _magma_data; break;
			}
		}
	}

	// Destructor
	ColorRamp::~ColorRamp(){
	}

	// Compute one HSV component
	void ColorRamp::rainbow(float v, float *cols){
		cols[0] = 1;
		cols[1] = 1;
		cols[2] = 1;

		if (v < 0.25f)
		{
			cols[0] = 0;
			cols[1] = 4 * v;
		}
		else if (v < 0.5f)
		{
			cols[0] = 0;
			cols[2] = 1 + 4 * (0.25f - v);
		}
		else if (v < 0.75f)
		{
			cols[0] = 4 * (v - 0.5f);
			cols[2] = 0;
		}
		else
		{
			cols[1] = 1 + 4 * (0.75f - v);
			cols[2] = 0;
		}
	}


	void ColorRamp::get_color(unsigned char i, ColorMap cmap, float *cols, bool inv){
		if (cmap != COLOR_NONE && cmap != COLOR_NUMBER)
		{
			if (inv) i = 255 - i;
			cols[0] = colormaps[cmap][3 * i];
			cols[1] = colormaps[cmap][3 * i + 1];
			cols[2] = colormaps[cmap][3 * i + 2];
		}
	}
	void ColorRamp::set_GLcolor(double v, ColorMap cmap, vec3 &color, bool inv){
		if (cmap != COLOR_NONE && cmap != COLOR_NUMBER)
		{
			int i = (int)((fabs(v)) * 255.0);//sqrt
			if (i > 255) i = 255;
			if (inv) i = 255 - i;
			color.x = colormaps[cmap][3 * i];
			color.y = colormaps[cmap][3 * i + 1];
			color.z = colormaps[cmap][3 * i + 2];
		}
	}

}