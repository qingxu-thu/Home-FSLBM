#pragma  once
#include <cmath>
#ifndef MFREE_MY_COLORRAMP_H
#define MFREE_MY_COLORRAMP_H
namespace Mfree
{
	struct vec3
	{
		float x;
		float y;
		float z;
		vec3(float a, float b, float c)
			: x{ a }
			, y{ b }
			, z{ c }
		{
		}

	};

	enum ColorMap
	{
		COLOR_NONE = 0,
		COLOR__MAGMA,
		COLOR_NUMBER

	};

	class ColorRamp
	{
	public:
		ColorRamp();
		~ColorRamp();

		void reset_random();

		void get_color(unsigned char i, ColorMap cmap, float* cols, bool inv = false);
		void set_GLcolor(double v, ColorMap cmap, vec3& color, bool inv = false);


	private:
		float color_random[256 * 3];
		float color_grayscale[256 * 3];
		float color_rainbow[256 * 3];
		float* colormaps[COLOR_NUMBER];
	private:
		static void rainbow(float v, float* col);

	};

}

#endif // !MFREE_MY_COLORRAMP_H
