// #include <bits/stdc++.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include "sharedmatting.h"

using namespace std;

vector <string> input_filename;
vector <string> trimap_filename;
vector <string> output_filename;
// vector <cv::Mat> images;

string input = "test/";
string trimap = "trimap/";
string output = "output/";
string ext = ".jpg";
string path = "/home/ap229997/carvana/dataset/"; // specify path accordingly

int main()
{
	ostringstream ss;
	
	cv::glob (path+input+"*"+ext, input_filename, false);
	size_t count = input_filename.size();

	// count = 20;

	for (int i=0; i<count; i++)
	{
		ss << i;
		trimap_filename.push_back(ss.str());
		output_filename.push_back(ss.str());
		ss.str("");
	}

	// cout<<fn[0]<<endl;
	/*
	for (size_t i=0; i<2; i++)
	{
		images.push_back(imread(fn[i]));
	}
	*/

	SharedMatting sm;

	for (int i=0; i<count; i++)
	{
		sm.loadImage(input_filename[i]);
		cout<<i<<" "<<path+trimap+"trimap"+trimap_filename[i]+ext<<endl;
		sm.loadTrimap(path+trimap+"trimap"+trimap_filename[i]+ext); // specify the path of Trimaps
		sm.solveAlpha();
		sm.save(path+output+"output"+output_filename[i]+ext); // specify the path of output images
	}
	

	return 0;
}