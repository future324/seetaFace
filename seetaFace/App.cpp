#include"Recognizer.hpp"
#define devView(i) imshow(#i,i);
int main()
{
	VideoCapture cam(0);
	if (!cam.isOpened()) return -1;
	// 模块初始化
	// 生成人脸比对数据
	// todo:把vector<Identity> IdentitiesLib写到内存里，下次直接载入，节约时间。
	
	// 运行
	Recongnizer face;
	face.Init();
	Mat frame; cam >> frame;
	for (; waitKey(1) != 27; cam >> frame)
	{
		face.Recongnize(frame);
		devView(frame);
	}
}

