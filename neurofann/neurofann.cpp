// neurofann.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <fstream>
#include <iostream>
#include "fann.h"
#include "floatfann.h"

using namespace std;

void txt_read(float* in, int in_number, string filename) {
	ifstream file(filename);
	int i = 0;
	for (;i < in_number;i++) {
		if (file.eof()) break;
		file >> in[i];
	}
	file.close();
	for (;i < in_number;i++) in[i] = 0;
}
struct bmpinfo {
	int size, pixels_adress, width, height;
	short int bits_per_pixel;
	bmpinfo() {
		size = 0;
		pixels_adress = 0;
		width = 0;
		height = 0;
		bits_per_pixel = 0;
	}
};
bmpinfo bmp_read(float* in, int in_number, string filename) {
	// Открываем файл
	ifstream file(filename, ios::in | ios::binary);

	bmpinfo info;

	// Переходим на 2 байт
	file.seekg(2, ios::beg);

	// Считываем размер файла
	file.read((char*)&info.size, sizeof(int));

	// Переходим на 10 байт
	file.seekg(10, ios::beg);

	// Считываем адрес начала массива пикселей
	file.read((char*)&info.pixels_adress, sizeof(int));

	file.seekg(18, ios::beg);

	// Считываем ширину и высоту изображения (в пикселях)
	file.read((char*)&info.width, sizeof(int));

	file.read((char*)&info.height, sizeof(int));


	file.seekg(28, ios::beg);

	// считываем кол-во битов на пиксель
	file.read((char*)&info.bits_per_pixel, sizeof(short int));

	file.seekg(info.pixels_adress, ios::beg);

	int i = 0;
	int offset = (32 - ((info.bits_per_pixel*info.width) % 32)) / 8;
	///////////////////// 1 BIT
	if (info.bits_per_pixel == 1)
	{
		unsigned char bgr;

		for (int y = info.height - 1; y >= 0; y--)
		{
			for (int x = 0; x < info.width; x++)
			{
				file.read((char*)&bgr, 1);

				if (x >= info.width - 4)
				{
					for (int n = 7; n >= 4; n--)
					{
						if (bgr & (1 << n)) // if 1 bit of readed pixels and 0b10000000
							in[i++] = 0;
						else
							in[i++] = 1;

						if (n != 4) x++;
					}
				}

				else
				{
					for (int n = 7; n >= 0; n--)
					{
						if (bgr & (1 << n)) // if 1 bit of readed pixels and 0b10000000
							in[i++] = 0;
						else
							in[i++] = 1;

						if (n != 0) x++;
					}
				}
			}
			file.read((char*)&bgr, offset); // offset
		}
	}
	//////////////// 1 bit END
	///////////////////// 4 BIT
	else if (info.bits_per_pixel == 4)
	{
		unsigned char bgr;

		for (int y = info.height - 1; y >= 0; y--)
		{
			for (int x = 0; x < info.width; x++)
			{

				file.read((char*)&bgr, 1);

				if (bgr & 0xF0)
					in[i++] = 0;
				else
					in[i++] = 1;

				x++;

				if (bgr & 0x0F)
					in[i++] = 0;
				else
					in[i++] = 1;
			}
			file.read((char*)&bgr, offset); // offset
		}
	}
	//////////////// 4 bit END
	///////////////////// 8 BIT
	else if (info.bits_per_pixel == 8)
	{
		unsigned char bgr;

		for (int y = info.height - 1; y >= 0; y--)
		{
			for (int x = 0; x < info.width; x++)
			{

				file.read((char*)&bgr, 1);

				if (bgr == 0xFF)
					in[i++] = 0;
				else
					in[i++] = 1;
			}
			file.read((char*)&bgr, offset); // offset
		}
	}
	//////////////// 8 bit END
	///////////////////// 16 BIT
	else if (info.bits_per_pixel == 16)
	{
		unsigned short int bgr;

		for (int y = info.height - 1; y >= 0; y--)
		{
			for (int x = 0; x < info.width; x++)
			{
				file.read((char*)&bgr, 2);

				if (bgr >= 0xFFF)
					in[i++] = 0;
				else
					in[i++] = 1;
			}
			file.read((char*)&bgr, offset); // offset
		}
	}
	//////////////// 16 bit END
	///////////////////// 24 BIT
	else if (info.bits_per_pixel == 24)
	{
		unsigned int bgr = 0;

		for (int y = info.height - 1; y >= 0; y--)
		{
			for (int x = 0; x < info.width; x++)
			{
				file.read((char*)&bgr, 3);

				//cout << bgr << endl;

				if (bgr == 0xFFFFFF)
					in[i++] = 0;
				else
					in[i++] = 1;

				bgr = 0;
			}
			file.read((char*)&bgr, offset); // offset
		}
	}
	//////////////// 24 bit END
	///////////////////// 32 BIT
	else if (info.bits_per_pixel == 32)
	{
		unsigned int bgr;

		for (int y = info.height - 1; y >= 0; y--)
		{
			for (int x = 0; x < info.width; x++)
			{
				file.read((char*)&bgr, 4);

				if (bgr >= 0xFFFFFF)
					in[i++] = 0;
				else
					in[i++] = 0;
			}
		}
	}
	//////////////// 32 bit END
	else
	{
		cerr << "Извините, Ваше изображение должно иметь 1, 4, 8, 16, 24 или 32 бит на пиксель. " << endl;
	}
	file.close();
	for (;i < in_number;i++) in[i] = 0;
	return info;
}

int main()
{
	unsigned int layers[4] = { 250000,1000,500,50 };
	struct fann* ann = fann_create_standard_array(4, layers);
	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	float* in=new float[layers[0]];
	bmp_read(in, layers[0], "big.bmp");

	//begin TIMECONTROL
	LARGE_INTEGER t1, t2, f;
	QueryPerformanceCounter(&t1);
	//end TIMECONTROL

	float* out = fann_run(ann, in);

	//begin TIMECONTROL
	QueryPerformanceCounter(&t2);
	QueryPerformanceFrequency(&f);
	double sec = double(t2.QuadPart - t1.QuadPart) / f.QuadPart * 1000.0;
	ofstream file;
	file.open("timecontrol.txt", ios::app);
	file << "calc => " << sec << "ms\n";
	file.close();
	//end TIMECONTROL

	//begin create training.txt
	file.open("training.txt");
	file << "1" << endl;
	for (int i = 0;i < layers[0];i++) file << in[i]<< " ";
	file << endl;
	for (int i = 0;i < layers[3];i++) file << out[i] << " ";
	file.close();
	//end create training.txt

	//begin TIMECONTROL
	QueryPerformanceCounter(&t1);
	//end TIMECONTROL

	fann_train_on_file(ann, "training.txt", 1, 1, 0.01);

	//begin TIMECONTROL
	QueryPerformanceCounter(&t2);
	QueryPerformanceFrequency(&f);
	sec = double(t2.QuadPart - t1.QuadPart) / f.QuadPart * 1000.0;
	file.open("timecontrol.txt", ios::app);
	file << "training => " << sec << "ms\n";
	file.close();
	//end TIMECONTROL

	file.open("out.txt");
	for (int i = 0;i < layers[3];i++) {
		file << out[i] << " ";
	}
	file.close();

	fann_save(ann, "language_classify.net");
	fann_destroy(ann);
	return 0;
}