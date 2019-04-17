
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cmath>
#include <string>
#include <fstream>

#define tr_speed 0.08
#define DEBUG
using namespace std;

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

class neuron {
	public:
		neuron(int in_number) {
			this->in_number = in_number;
			weight = new int[in_number];
			for (int i = 0;i < in_number;i++) 
				weight[i] = rand() % 200 - 100;
		}
		neuron(string filename = "weight_neuron.dat") {
			open(filename);
		}
		neuron(int in_number, int* weigths) {
			this->in_number = in_number;
			weight = new int[in_number];
			for (int i = 0;i < in_number;i++) weight[i] = weigths[i];
		}
		~neuron() {
			delete weight;
		}
		neuron& operator=(const neuron& right) {
			//проверка на самоприсваивание
			if (this == &right) {
				return *this;
			}
			in_number = right.in_number;
			weight = new int[in_number];
			for (int i = 0;i < in_number;i++) weight[i] = right.weight[i];
			return *this;
		}
		neuron(const neuron & object)
		{
			in_number = object.in_number;
			weight = new int[in_number];
			for (int i = 0;i < in_number;i++) weight[i] = object.weight[i];
		}
		void save(string filename = "weight_neuron.dat") {
			ofstream file(filename);
			file << this->in_number << " ";
			weight = new int[in_number];
			for (int i = 0;i < in_number;i++) {
				file << weight[i] << " ";
			}
			file.close();
		}
		void open(string filename = "weight_neuron.dat") {
			ifstream file(filename);
			file >> this->in_number;
			int i = 0;
			for (;i < in_number;i++) {
				if (file.eof()) break;
				file >> weight[i];
			}
			file.close();
			srand(time(0)); // автоматическа€ рандомизаци€
			for (;i < in_number;i++) weight[i] = rand() % 200 - 100;
		}
		int calc(int* in) {
			int sum = 0;
			for (int i = 0;i < in_number;i++) sum += in[i] * weight[i];
			return threshold(sum);
		}
		int threshold(int arg) {
			/*//ѕорогова€ функци€1000 * in_number*0.6
			if (arg > 0) return 1;
			else return 0;*/
			/*//—игмоида
			//1 / (1 + Exp[-(x - 1000 * 10 / 2) / 1000])
			//1000.0 / (1.0 + Exp[-x / 100])
			double res = 1000.0 / (1.0+exp((double)(-arg)/((double)in_number*100.0)));
			return (int)res;*/
			//Ћинейна€ функци€
			return arg;
		}
		int derivative(int arg) {
			/*//Ћинейной функции с насыщением
			if (arg > -1000 * in_number*0.3 & arg < 1000 * in_number*0.3) return 1;
			else return 0;*/
			/*//—игмоиды
			double res = 100.0*1000.0*exp(-100.0*arg/ (double)in_number)/ (double)in_number/(1.0+ exp(-100.0*arg / (double)in_number));
			return (int)res;*/
			//Ћинейной функции
			return 1;
		}
		int* weight;
		int in_number;
};

class layer {
public:	
	layer(int in_number, int neuro_number) {
		this->neuro_number = neuro_number;
		neurons = (neuron*)malloc(sizeof(neuron)*neuro_number);
		for (int i = 0;i < neuro_number;i++) 
			neurons[i] = neuron(in_number);
	}
	layer(string filename = "weight_layer.dat") {
		open(filename);
	}
	layer(int in_number, int neuro_number, int** weights) {
		this->neuro_number = neuro_number;
		neurons = (neuron*)malloc(sizeof(neuron)*neuro_number);
		//for (int i = 0;i < neuro_number;i++) for (int j = 0;j < in_number;j++) neurons[i].weight[j]=weights[i][j];
		for (int i = 0;i < neuro_number;i++) neurons[i] = neuron(in_number, weights[i]);
	}
	layer& operator=(const layer& right) {
		//проверка на самоприсваивание
		if (this == &right) {
			return *this;
		}
		neuro_number = right.neuro_number;
		neurons = new neuron[neuro_number];
		for (int i = 0;i < neuro_number;i++) neurons[i] = right.neurons[i];
		return *this;
	}
	layer(const layer & object)
	{
		neuro_number = object.neuro_number;
		neurons = new neuron[neuro_number];
		for (int i = 0;i < neuro_number;i++) neurons[i] = object.neurons[i];
	}
	void open(string filename = "weight_layer.dat") {
		ifstream file(filename);
		file>>this->neuro_number;
		int in_number;
		file >> in_number;
		neurons = (neuron*)malloc(sizeof(neuron)*neuro_number);
		for (int i = 0;i < neuro_number;i++) neurons[i] = neuron(in_number);
		for (int i = 0;i < neuro_number;i++) for (int j = 0;j < in_number;j++) {
			if (file.eof()) break;
			file >> neurons[i].weight[j];
		}
		file.close();
	}
	void save(string filename = "weight_layer.dat") {
		int in_number = neurons[0].in_number;
		ofstream file(filename);
		file << this->neuro_number << " ";
		file << in_number << " ";
		for (int i = 0;i < neuro_number;i++) for (int j = 0;j < in_number;j++) {
			file << neurons[i].weight[j] << " ";
		}
		file.close();
	}
	~layer() {
		delete neurons;
	}
	int* calc(int* in) {
		int* result = new int[neuro_number];
		for (int i = 0;i < neuro_number;i++) result[i] = neurons[i].calc(in);
		return result;
	}
	/*void training(int* outerror) {
		int* out_real = calc(in);

	}*/
	int neuro_number;
	neuron* neurons;
};

class net {
public:
	net(int* in_number, int layers_number) {
		this->layers_number = layers_number;
		layers = (layer*)(malloc(sizeof(layer)*layers_number));
		for (int i = 0;i < layers_number;i++) 
			layers[i] = layer(in_number[i], in_number[i+1]);
	}
	net(string filename = "weight_net.dat") {
		open(filename);
	}
	net(int* in_number, int layers_number, int*** weights) {
		this->layers_number = layers_number;
		layers = (layer*)(malloc(sizeof(layer)*layers_number));
		for (int i = 0;i < layers_number;i++) layers[i] = layer(in_number[i], in_number[i + 1],weights[i]);
	}
	net& operator=(const net& right) {
		//проверка на самоприсваивание
		if (this == &right) {
			return *this;
		}
		layers_number = right.layers_number;
		layers = new layer[layers_number];
		for (int i = 0;i < layers_number;i++) layers[i] = right.layers[i];
		return *this;
	}
	net(const net & object)
	{
		layers_number = object.layers_number;
		layers = new layer[layers_number];
		for (int i = 0;i < layers_number;i++) layers[i] = object.layers[i];
	}
	void open(string filename = "weight_net.dat") {
		ifstream file(filename);
		file >> this->layers_number;
		layers = (layer*)(malloc(sizeof(layer)*layers_number));
		int* in_number = new int[layers_number+1];
		for (int i = 0;i < layers_number + 1;i++) file >> in_number[i];
		for (int i = 0;i < layers_number;i++) 
			layers[i] = layer(in_number[i], in_number[i + 1]);
		for (int z = 0;z < layers_number;z++) {
			int in_number = layers[z].neurons[0].in_number;
			for (int i = 0;i < layers[z].neuro_number;i++) for (int j = 0;j < in_number;j++) {
				if (file.eof()) break;
				file >> layers[z].neurons[i].weight[j];
			}
		}
		file.close();
	}
	void save(string filename = "weight_net.dat") {
		ofstream file(filename);
		file << this->layers_number << " ";
		for (int z = 0;z < layers_number;z++) file << layers[z].neurons[0].in_number << " ";
		file << layers[layers_number-1].neuro_number << " ";
		for (int z = 0;z < layers_number;z++) {
			int in_number = layers[z].neurons[0].in_number;
			for (int i = 0;i < layers[z].neuro_number;i++) for (int j = 0;j < in_number;j++) {
				file << layers[z].neurons[i].weight[j]<<" ";
			}
		}
		file.close();
	}
	~net() {
		delete layers;
	}
	int* calc(int* in) {
		int* result=in;
		for (int i = 0;i < layers_number;i++) {
			result = layers[i].calc(result);
		}
		return result;
	}
	void training(int* innet, int* outnet) {

		int** net;
		int** y;
		net = new int*[layers_number];
		y = new int*[layers_number];

		y[0] = new int[layers[0].neuro_number];
		net[0] = new int[layers[0].neuro_number];
		for (int j = 0;j < layers[0].neuro_number;j++) {
			int sum = 0;
			for (int k = 0;k < layers[0].neurons[j].in_number;k++) sum += layers[0].neurons[j].weight[k] * innet[k];
			y[0][j] = layers[0].neurons[j].threshold(sum);
			net[0][j] = sum;
		}

		for (int i = 1;i < layers_number;i++) {
			y[i] = new int[layers[i].neuro_number];
			net[i] = new int[layers[i].neuro_number];
			for (int j = 0;j < layers[i].neuro_number;j++) {
				int sum = 0;
				for (int k = 0;k < layers[i].neurons[j].in_number;k++) sum += layers[i].neurons[j].weight[k] * y[i - 1][k];
				y[i][j] = layers[i].neurons[j].threshold(sum);
				net[i][j] = sum;
			}
		}

		int** g;
		g = new int*[layers_number];
		g[layers_number - 1] = new int[layers[layers_number - 1].neuro_number];
		for (int j = 0;j < layers[layers_number - 1].neuro_number;j++) g[layers_number - 1][j] = 2 * (outnet[j] - y[layers_number - 1][j]) * layers[layers_number - 1].neurons[j].derivative(net[layers_number - 1][j]);
		for (int i = layers_number - 2;i >= 0;i--) {
			g[i] = new int[layers[i].neuro_number];
			for (int j = 0;j < layers[i].neuro_number;j++) {
				g[i][j] = 0;
				for (int k = 0;k < layers[i + 1].neuro_number;k++) g[i][j] += g[i + 1][k] * layers[i + 1].neurons[k].weight[j];
				g[i][j] *= layers[i].neurons[j].derivative(net[i][j]);
			}
		}
		int*** derr;
		derr = new int**[layers_number];
		derr[0] = new int*[layers[0].neuro_number];
		for (int j = 0;j < layers[0].neuro_number;j++) {
			int a = layers[0].neurons[j].in_number;
			derr[0][j] = new int[a];
			for (int k = 0;k < layers[0].neurons[j].in_number;k++)
				derr[0][j][k] = g[0][j] * innet[k];
		}
		for (int i = 1;i < layers_number;i++) {
			derr[i] = new int*[layers[i].neuro_number];
			for (int j = 0;j < layers[i].neuro_number;j++) {
				derr[i][j] = new int[layers[i].neurons[j].in_number];
				for (int k = 0;k < layers[i].neurons[j].in_number;k++)
					derr[i][j][k] = g[i][j] * y[i - 1][k];
			}
		}
		for (int i = 0;i < layers_number;i++)
			for (int j = 0;j < layers[i].neuro_number;j++)
				for (int k = 0;k < layers[i].neurons[j].in_number;k++) {
					int dr = derr[i][j][k];
					layers[i].neurons[j].weight[k] -= dr*tr_speed;
				}
		
	}
	int layers_number;
	layer* layers;
};

void txt_read(int* in, int in_number, string filename) {
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
	int size = 0, pixels_adress = 0, width = 0, height = 0;
	short int bits_per_pixel = 0;
};
bmpinfo bmp_read(int* in, int in_number, string filename) {
	// ќткрываем файл
	ifstream file(filename, ios::in | ios::binary);

	bmpinfo info;

	// ѕереходим на 2 байт
	file.seekg(2, ios::beg);

	// —читываем размер файла
	file.read((char*)&info.size, sizeof(int));

	// ѕереходим на 10 байт
	file.seekg(10, ios::beg);

	// —читываем адрес начала массива пикселей
	file.read((char*)&info.pixels_adress, sizeof(int));

	file.seekg(18, ios::beg);

	// —читываем ширину и высоту изображени€ (в пиксел€х)
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

		for (int y = info.height-1; y >= 0; y--)
		{
			for (int x = 0; x < info.width; x++)
			{
				file.read((char*)&bgr, 1);

				if (x >= info.width-4)
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

		for (int y = info.height-1; y >= 0; y--)
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

		for (int y = info.height-1; y >= 0; y--)
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

		for (int y = info.height-1; y >= 0; y--)
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

		for (int y = info.height-1; y >= 0; y--)
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

		for (int y = info.height-1; y >= 0; y--)
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
		cout << "»звините, ¬аше изображение должно иметь 1, 4, 8, 16, 24 или 32 бит на пиксель. " << endl;
	}
	file.close();
	for (;i < in_number;i++) in[i] = 0;
	return info;
}

/*__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}*/

int main()
{
	srand(time(0)); // автоматическа€ рандомизаци€
/*    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }*/

	string comand="";
	bool training_mode = false;
	bool bmp_mode = false;
	string infile="in.txt";
	string outfile = "out.txt";
	setlocale(LC_ALL, "Russian");
	cout << "¬ведите команду:\n\"file\" - загрузить нейросеть из файла\n\"new\" - создать новую нейросеть\n\"exit\" - выйти\n";
	net* neuro;
	while (comand != "exit") {
		cin >> comand;
#ifdef DEBUG
		cout <<"¬ведено в цикле 1: "<< comand<<"\n";
#endif
		if (comand == "exit") break;
		if (comand == "file") {
			string file = "";
			cout << "¬ведите адрес файла:\n";
			cin >> file;
#ifdef DEBUG
			cout << "¬ведено: " << file << "\n";
#endif
			neuro = new net(file);
			cout << "OK\n";
			break;
		}
		if (comand == "new") {
			int layers_number;
			cout << "¬ведите количество слоЄв:\n";
			cin >> layers_number;
			int* in_number;
			in_number = new int[layers_number + 1];
			cout << "¬ведите количество входов сети:\n";
			cin >> in_number[0];
			for (int i = 1;i < layers_number + 1;i++) {
				cout << "¬ведите количество нейронов "<< i <<"-го сло€:\n";
				cin >> in_number[i];
			}
			neuro = new net(in_number, layers_number);
			cout << "OK\n";
			break;
		}
	}
	cout << "¬ведите команду:\n\"normal\" - перейти в рабочий режим (по умолчанию)\n\"training\" - перейти в режим обучени€\n";
	cout << "\"txt\" - указать входной текстовый файл (по умолчанию in.txt)\n\"bmp\" - указать входной *.bmp файл\n";
	cout << "\"out\" - указать выходной текстовый файл (по умолчанию out.txt)\n";
	cout << "\"run\" - запустить\n\"save\" - сохранить сеть в файл\n";
	cout << "\"open\" - загрузить сеть из файла\n\"exit\" - выйти\n";
	while (comand != "exit") {
		cin >> comand;
#ifdef DEBUG
		cout << "¬ведено в цикле 2: " << comand << "\n";
#endif
		if (comand == "exit") break;
		if (comand == "normal") training_mode = false;
		if (comand == "training") training_mode = true;
		if (comand == "txt") {
			bmp_mode = false;
			cout << "¬ведите адрес файла:\n";
			cin >> infile;
#ifdef DEBUG
			cout << "¬ведено: " << infile << "\n";
#endif
			cout << "OK\n";
		}
		if (comand == "bmp") {
			bmp_mode = true;
			cout << "¬ведите адрес файла:\n";
			cin >> infile;
#ifdef DEBUG
			cout << "¬ведено: " << infile << "\n";
#endif
			cout << "OK\n";
		}
		if (comand == "out") {
			cout << "¬ведите адрес файла:\n";
			cin >> outfile;
#ifdef DEBUG
			cout << "¬ведено: " << outfile << "\n";
#endif
			cout << "OK\n";
		}
		if (comand == "save") {
			string file = "";
			cout << "¬ведите адрес файла:\n";
			cin >> file;
#ifdef DEBUG
			cout << "¬ведено: " << file << "\n";
#endif
			neuro->save(file);
			cout << "OK\n";
		}
		if (comand == "open") {
			string file = "";
			cout << "¬ведите адрес файла:\n";
			cin >> file;
#ifdef DEBUG
			cout << "¬ведено: " << file << "\n";
#endif
			neuro->open(file);
			cout << "OK\n";
		}
		if (comand == "run") {
			int* in=new int[neuro->layers[0].neurons[0].in_number];
			if (!bmp_mode) {
				txt_read(in, neuro->layers[0].neurons[0].in_number, infile);	
			}
			else {
				bmpinfo info=bmp_read(in, neuro->layers[0].neurons[0].in_number, infile);
#ifdef DEBUG
				cout << "Bits per pixel: "<<info.bits_per_pixel<<"\n";
				cout << "Height: " << info.height << "\n";
				cout << "Width: " << info.width << "\n";
				cout << "Pixels adress: " << info.pixels_adress << "\n";
				cout << "Size: " << info.size << "\n";
#endif
			}
#ifdef DEBUG
			ofstream file("inctrl.txt");
			for (int i = 0;i < neuro->layers[0].neurons[0].in_number;i++) {
				file << in[i] << " ";
			}
			file.close();
#endif
			if (!training_mode) {
				int* out=neuro->calc(in);
				ofstream file(outfile);
				for (int i = 0;i < neuro->layers[neuro->layers_number-1].neuro_number;i++) {
					file << out[i] << " ";
				}
				file.close();
			}else{
				ifstream file(outfile);
				int* out = new int[neuro->layers[neuro->layers_number-1].neuro_number];
				int i = 0;
				for (;i < neuro->layers[neuro->layers_number-1].neuro_number;i++) {
					if (file.eof()) break;
					file >> out[i];
				}
				file.close();
				for (;i < neuro->layers[neuro->layers_number-1].neuro_number;i++) out[i] = 0;

#ifdef DEBUG
				ofstream file2("outctrl.txt");
				for (int i = 0;i < neuro->layers[neuro->layers_number - 1].neuro_number;i++) {
					file2 << out[i] << " ";
				}
				file2.close();
#endif
				neuro->training(in, out);
			}
			cout << "OK\n";
		}
	}

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
/*cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}*/
