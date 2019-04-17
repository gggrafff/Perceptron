
#include <iostream>
#include <ctime>
#include <cmath>
#include <string>
#include <fstream>
#include <windows.h>

#define tr_speed 0.02f
//#define DEBUG
#define TIMECONTROL
using namespace std;

class neuron {
public:
	neuron(int in_number) {
		this->in_number = in_number;
		weight = new float[in_number];
		for (int i = 0;i < in_number;i++)
			weight[i] = (float)(rand() % 200 - 100)/1000.0f;
	}
	neuron(string filename = "weight_neuron.dat") {
		open(filename);
	}
	neuron(int in_number, float* weigths) {
		this->in_number = in_number;
		weight = new float[in_number];
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
		weight = new float[in_number];
		for (int i = 0;i < in_number;i++) weight[i] = right.weight[i];
		return *this;
	}
	neuron(const neuron & object)
	{
		in_number = object.in_number;
		weight = new float[in_number];
		for (int i = 0;i < in_number;i++) weight[i] = object.weight[i];
	}
	void save(string filename = "weight_neuron.dat") {
		ofstream file(filename);
		file << this->in_number << " ";
		weight = new float[in_number];
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
		//srand(time(0)); // автоматическа€ рандомизаци€
		for (;i < in_number;i++) weight[i] = (float)(rand() % 200 - 100) / 1000.0f;
	}
	float calc(float* in) {
		float sum = 0;
		for (int i = 0;i < in_number;i++) sum += in[i] * weight[i];
		return threshold(sum);
	}
	float threshold(float arg) {
		//—игмоида
		float res = 1.0f / (1.0f+exp(-arg));
		return res;
		//Ћинейна€ функци€
		//return arg;
	}
	float derivative(float arg) {
		//—игмоиды
		float res = exp(arg)/ pow((1.0f+ exp(arg)),2);
		return res;
		//Ћинейной функции
		//return 1;
	}
	float* weight;
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
	layer(int in_number, int neuro_number, float** weights) {
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
		file >> this->neuro_number;
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
	float* calc(float* in) {
		float* result = new float[neuro_number];
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
			layers[i] = layer(in_number[i], in_number[i + 1]);
	}
	net(string filename, bool txt_mode) {
		open(filename,txt_mode);
	}
	net(int* in_number, int layers_number, float*** weights) {
		this->layers_number = layers_number;
		layers = (layer*)(malloc(sizeof(layer)*layers_number));
		for (int i = 0;i < layers_number;i++) layers[i] = layer(in_number[i], in_number[i + 1], weights[i]);
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
	void open(string filename, bool txt_mode) {
		if (txt_mode) {
			ifstream file(filename);
			file >> this->layers_number;
			layers = (layer*)(malloc(sizeof(layer)*layers_number));
			int* in_number = new int[layers_number + 1];
			for (int i = 0;i < layers_number + 1;i++) file >> in_number[i];
			for (int i = 0;i < layers_number;i++)
				layers[i] = layer(in_number[i], in_number[i + 1]);

			for (int z = 0;z < layers_number;z++) {
				int in_number = layers[z].neurons[0].in_number;
				for (int i = 0;i < layers[z].neuro_number;i++) for (int j = 0;j < in_number;j++) {
					file >> layers[z].neurons[i].weight[j];
				}
			}
			file.close();
		}
		else {
			ifstream file(filename, ios::binary);
			file.read((char*)&(this->layers_number), sizeof(this->layers_number));
			layers = (layer*)(malloc(sizeof(layer)*layers_number));
			int* in_number = new int[layers_number + 1];
			file.read((char*)in_number, sizeof(int)*(layers_number + 1));
			for (int i = 0;i < layers_number;i++)
				layers[i] = layer(in_number[i], in_number[i + 1]);
			for (int z = 0;z < layers_number;z++) {
				int in_number = layers[z].neurons[0].in_number;
				//cout << "in_number=" << in_number;
				//cout << "neuro_number=" << layers[z].neuro_number;
				for (int i = 0;i < layers[z].neuro_number;i++) {
					file.read((char*)layers[z].neurons[i].weight, sizeof(float)*in_number);
				}
			}
			file.close();
		}

	}
	void save(string filename, bool txt_mode) {
		if (txt_mode) {
		ofstream file(filename);
		file << this->layers_number << " ";
		for (int z = 0;z < layers_number;z++) file << layers[z].neurons[0].in_number << " ";
		file << layers[layers_number - 1].neuro_number << " ";

		for (int z = 0;z < layers_number;z++) {
			int in_number = layers[z].neurons[0].in_number;
			for (int i = 0;i < layers[z].neuro_number;i++) for (int j = 0;j < in_number;j++) {
				file << layers[z].neurons[i].weight[j] << " ";
			}
		}
		file.close();
		}
		else {
			ofstream file(filename, ios::binary);
			file.write((char*)&(this->layers_number), sizeof(this->layers_number));
			for (int z = 0;z < layers_number;z++) 
				file.write((char*)&(layers[z].neurons[0].in_number), sizeof(int));
			file.write((char*)&(layers[layers_number - 1].neuro_number), sizeof(int));

			for (int z = 0;z < layers_number;z++) {
			int in_number = layers[z].neurons[0].in_number;
			for (int i = 0;i < layers[z].neuro_number;i++) {
				file.write((char*)layers[z].neurons[i].weight, sizeof(float)*in_number);
			}
			}
			file.close();
		}
	}
	~net() {
		delete layers;
	}
	float* calc(float* in) {
#ifdef TIMECONTROL
		LARGE_INTEGER t1, t2, f;
		QueryPerformanceCounter(&t1);
#endif

		float* result = in;
		for (int i = 0;i < layers_number;i++) {
			result = layers[i].calc(result);
		}

#ifdef TIMECONTROL
		QueryPerformanceCounter(&t2);
		QueryPerformanceFrequency(&f);
		double sec = double(t2.QuadPart - t1.QuadPart) / f.QuadPart * 1000.0;
		ofstream file;
		file.open("timecontrol.txt", ios::app);
		file << "calc => " << sec << "ms\n";
		file.close();
#endif
		return result;
	}
	void training(float* innet, float* outnet) {
#if (defined TIMECONTROL) || (defined DEBUG)
		ofstream file;
#endif


#ifdef TIMECONTROL
		LARGE_INTEGER t1, t2, f;
		QueryPerformanceCounter(&t1);
#endif
		float** net;
		float** y;
		net = new float*[layers_number];
		y = new float*[layers_number];

		y[0] = new float[layers[0].neuro_number];
		net[0] = new float[layers[0].neuro_number];
		for (int j = 0;j < layers[0].neuro_number;j++) {
			float sum = 0;
			for (int k = 0;k < layers[0].neurons[j].in_number;k++) sum += layers[0].neurons[j].weight[k] * innet[k];
			y[0][j] = layers[0].neurons[j].threshold(sum);
			net[0][j] = sum;
		}

		for (int i = 1;i < layers_number;i++) {
			y[i] = new float[layers[i].neuro_number];
			net[i] = new float[layers[i].neuro_number];
			for (int j = 0;j < layers[i].neuro_number;j++) {
				float sum = 0;
				for (int k = 0;k < layers[i].neurons[j].in_number;k++) sum += layers[i].neurons[j].weight[k] * y[i - 1][k];
				y[i][j] = layers[i].neurons[j].threshold(sum);
				net[i][j] = sum;
			}
		}

#ifdef DEBUG
		file.open("trainingnetctrl.txt");
		for (int i = 0;i < layers_number;i++) {
			for (int j = 0;j < layers[i].neuro_number;j++) {
					file << net[i][j] << " ";
			}
		}
		file.close();
		file.open("trainingyctrl.txt");
		for (int i = 0;i < layers_number;i++) {
			for (int j = 0;j < layers[i].neuro_number;j++) {
				file << y[i][j] << " ";
			}
		}
		file.close();
#endif

		float** g;
		g = new float*[layers_number];
		g[layers_number - 1] = new float[layers[layers_number - 1].neuro_number];
		for (int j = 0;j < layers[layers_number - 1].neuro_number;j++) g[layers_number - 1][j] = 2 * (outnet[j] - y[layers_number - 1][j]) * layers[layers_number - 1].neurons[j].derivative(net[layers_number - 1][j]);
		for (int i = layers_number - 2;i >= 0;i--) {
			g[i] = new float[layers[i].neuro_number];
			for (int j = 0;j < layers[i].neuro_number;j++) {
				g[i][j] = 0;
				for (int k = 0;k < layers[i + 1].neuro_number;k++) g[i][j] += g[i + 1][k] * layers[i + 1].neurons[k].weight[j];
				g[i][j] *= layers[i].neurons[j].derivative(net[i][j]);
			}
		}

#ifdef DEBUG
		file.open("traininggctrl.txt");
		for (int i = 0;i < layers_number;i++) {
			for (int j = 0;j < layers[i].neuro_number;j++) {
				file << g[i][j] << " ";
			}
		}
		file.close();
#endif

		float*** derr;
		derr = new float**[layers_number];
		derr[0] = new float*[layers[0].neuro_number];
		for (int j = 0;j < layers[0].neuro_number;j++) {
			int a = layers[0].neurons[j].in_number;
			derr[0][j] = new float[a];
			for (int k = 0;k < layers[0].neurons[j].in_number;k++)
				derr[0][j][k] = g[0][j] * innet[k];
		}
		for (int i = 1;i < layers_number;i++) {
			derr[i] = new float*[layers[i].neuro_number];
			for (int j = 0;j < layers[i].neuro_number;j++) {
				derr[i][j] = new float[layers[i].neurons[j].in_number];
				for (int k = 0;k < layers[i].neurons[j].in_number;k++)
					derr[i][j][k] = g[i][j] * y[i - 1][k];
			}
		}
		for (int i = 0;i < layers_number;i++)
			for (int j = 0;j < layers[i].neuro_number;j++)
				for (int k = 0;k < layers[i].neurons[j].in_number;k++) {
					float dr = derr[i][j][k];
					layers[i].neurons[j].weight[k] += dr*tr_speed;
				}
#ifdef TIMECONTROL
		QueryPerformanceCounter(&t2);
		QueryPerformanceFrequency(&f);
		double sec = double(t2.QuadPart - t1.QuadPart) / f.QuadPart * 1000.0;
		file.open("timecontrol.txt", ios::app);
		file << "training => " << sec << "ms\n";
		file.close();
#endif
	}
	int layers_number;
	layer* layers;
};

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
	bmpinfo(){
		size = 0;
		pixels_adress = 0;
		width = 0;
		height = 0;
		bits_per_pixel = 0;
	}
};
bmpinfo bmp_read(float* in, int in_number, string filename) {
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
		cout << "»звините, ¬аше изображение должно иметь 1, 4, 8, 16, 24 или 32 бит на пиксель. " << endl;
	}
	file.close();
	for (;i < in_number;i++) in[i] = 0;
	return info;
}


int main()
{
	srand(time(0)); // автоматическа€ рандомизаци€

	string comand = "";
	bool training_mode = false;
	bool bmp_mode = false;
	bool txt_mode = false;
	string infile = "in.txt";
	string outfile = "out.txt";
	setlocale(LC_ALL, "Russian");
	cout << "¬ерси€ дл€ CPU.\n¬ведите команду:\n\"file\" - загрузить нейросеть из файла\n\"txt\" - установить тип входного файла - текстовый\n\"bin\" - установить тип входного файла - двоичный\n\"new\" - создать новую нейросеть\n\"exit\" - выйти\n";
	net* neuro=NULL;
	while (comand != "exit") {
		cin >> comand;
#ifdef DEBUG
		cout << "¬ведено в цикле 1: " << comand << "\n";
#endif
		if (comand == "exit") break;
		if (comand == "file") {
			string file = "";
			cout << "¬ведите адрес файла:\n";
			cin >> file;
#ifdef DEBUG
			cout << "¬ведено: " << file << "\n";
#endif
			neuro = new net(file,txt_mode);
			cout << "OK\n";
			break;
		}
		if (comand == "txt") {
			txt_mode = true;
			cout << "OK\n";
		}
		if (comand == "bin") {
			txt_mode = false;
			cout << "OK\n";
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
				cout << "¬ведите количество нейронов " << i << "-го сло€:\n";
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
		if (comand == "normal") {training_mode = false; cout << "OK\n";}
		if (comand == "training") {training_mode = true; cout << "OK\n";}
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
			neuro->save(file,txt_mode);
			cout << "OK\n";
		}
		if (comand == "open") {
			string file = "";
			cout << "¬ведите адрес файла:\n";
			cin >> file;
#ifdef DEBUG
			cout << "¬ведено: " << file << "\n";
#endif
			neuro->open(file,txt_mode);
			cout << "OK\n";
		}
		if (comand == "run") {
			float* in = new float[neuro->layers[0].neurons[0].in_number];
			if (!bmp_mode) {
				txt_read(in, neuro->layers[0].neurons[0].in_number, infile);
			}
			else {
				bmpinfo info = bmp_read(in, neuro->layers[0].neurons[0].in_number, infile);
#ifdef DEBUG
				cout << "Bits per pixel: " << info.bits_per_pixel << "\n";
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
				float* out = neuro->calc(in);
				ofstream file(outfile);
				for (int i = 0;i < neuro->layers[neuro->layers_number - 1].neuro_number;i++) {
					file << out[i] << " ";
				}
				file.close();
			}
			else {
				ifstream file(outfile);
				float* out = new float[neuro->layers[neuro->layers_number - 1].neuro_number];
				int i = 0;
				for (;i < neuro->layers[neuro->layers_number - 1].neuro_number;i++) {
					if (file.eof()) break;
					file >> out[i];
				}
				file.close();
				for (;i < neuro->layers[neuro->layers_number - 1].neuro_number;i++) out[i] = 0;

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
