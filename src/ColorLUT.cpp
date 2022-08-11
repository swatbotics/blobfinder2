#include "ColorLUT.h"
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <assert.h>

enum {
  ycount = (1 << ColorLUT::ybits),
  ccount = (1 << ColorLUT::cbits),
  arraysize = ycount*ccount*ccount
};

typedef ColorLUT::pixel pixel;

ColorLUT::ColorLUT() {
  lutdata.resize(arraysize, 0);
}

static void downsample(pixel& YCrCb) {

  // these guys are all 8 bits:
  unsigned char& Y  = YCrCb[2];
  unsigned char& Cr = YCrCb[1];
  unsigned char& Cb = YCrCb[0];
  /*Channels are downsampled by right-shifting 8 minus number of bits so that
  the value has ybits or cbits places*/
  Y =  (Y  >> (8-ColorLUT::ybits));
  Cr = (Cr >> (8-ColorLUT::cbits));
  Cb = (Cb >> (8-ColorLUT::cbits));

}

static size_t downsample2index(const pixel& YCrCb) {

  size_t Y  = YCrCb[2];
  size_t Cr = YCrCb[1];
  size_t Cb = YCrCb[0];
  /*Concatenate the values in the form YCrCb by bit-shifting each value to make
  room for the next and ORing them together	*/

  return ( (Y << (ColorLUT::cbits + ColorLUT::cbits)) | 
	   (Cr << ColorLUT::cbits) | 
	   Cb );

}

static size_t pixel2index(pixel YCrCb) {

  downsample(YCrCb);
  return downsample2index(YCrCb);

}



void ColorLUT::save(const std::string& filename) const {
  
  std::ofstream ostr(filename.c_str());
  if(!ostr.is_open()){
    throw std::runtime_error("File did not open!");
  }

  ostr << "ColorLUT\n"
       << ybits << "\n"
       << cbits << "\n"
       << numcolors << "\n";

  for (size_t i=0; i<numcolors; ++i) {
    ostr << colornames[i] << "\n";
  }

  ostr.write( (char*)(&lutdata[0]), 
	      sizeof(colorflags)*lutdata.size() );

}

void ColorLUT::addToColor(const pixel& YCrCb, 
			  size_t cidx, 
			  int yRange, int cRange) {
  
  /*NOTE: cidx is the index of the color in the color vector, index is the
    index of the flag in the lookup table*/
  pixel ds = YCrCb;
  downsample(ds);  //Downsample the channels of ds to byte form
  
  colorflags flag = (1 << cidx);
  
  for (int dy=-yRange; dy<=yRange; ++dy) {
    for (int dcr=-cRange; dcr<=cRange; ++dcr) {
      for (int dcb=-cRange; dcb<=cRange; ++dcb) {
	
	int y = ds[2] + dy;
	int cr = ds[1] + dcr;
	int cb = ds[0] + dcb;
        //If values are in the correct range to convert to index 
	if (y >= 0 && y < ycount &&
	    cr >= 0 && cr < ccount &&
	    cb >= 0 && cb < ccount) {
	  
	  pixel ds2(cb,cr,y);
	  size_t index = downsample2index(ds2);
          /*Indicate the index in the LUT data is associated with the color by
	    flipping the bit corresponding with the color flag*/
	  lutdata[index] = (lutdata[index] | flag);
	  
	}
	
      }
    }
  }
}

void ColorLUT::removeFromColor(const pixel& YCrCb, 
			       size_t cidx, 
			       int yRange, int cRange) {
  
  pixel ds = YCrCb;
  downsample(ds);

  colorflags flag = (1 << cidx);

  for (int dy=-yRange; dy<=yRange; ++dy) {
    for (int dcr=-cRange; dcr<=cRange; ++dcr) {
      for (int dcb=-cRange; dcb<=cRange; ++dcb) {
	
	int y = ds[2] + dy;
	int cr = ds[1] + dcr;
	int cb = ds[0] + dcb;
	
	if (y >= 0 && y < ycount &&
	    cr >= 0 && cr < ccount &&
	    cb >= 0 && cb < ccount) {
	  
	  pixel ds2(cb,cr,y);
	  size_t index = downsample2index(ds2);
          /*Clear the data at index by ANDing it with the negation of the
	    colorflag we want to clear*/
	  lutdata[index] = (lutdata[index] & ~flag);
	  
	}
	
      }
    }
  }



}

void ColorLUT::clearColor(size_t cidx) {

  colorflags clear = ~(1 << cidx);
  
  for (size_t i=0; i<arraysize; ++i) {
    lutdata[i] = lutdata[i] & clear;
  }
  
}

ColorLUT::colorflags 
ColorLUT::getColors(const pixel& YCrCb) const {
  
  size_t index = pixel2index(YCrCb);
  return lutdata[index];

}


size_t ColorLUT::addColor(const std::string& name) {
  for (size_t i=0; i<numcolors; ++i) {
    if (colornames[i] == "") {
      colornames[i] = name;
      return i;
    }
  }
  return npos;
}

void ColorLUT::removeColor(size_t cidx) {
  colornames[cidx] = "";
  clearColor(cidx);
}

size_t ColorLUT::lookupColor(const std::string& name) const {
  for (size_t i=0; i<numcolors; ++i) {
    if (colornames[i] == name) {
      return i;
    }
  }
  return npos;
}

static std::string int2str(int i) {
  std::ostringstream ostr;
  ostr << i;
  return ostr.str();
}

void ColorLUT::load(const std::string& filename) {

  std::ifstream istr(filename.c_str());

  if(!istr.is_open()){
    throw std::runtime_error("File did not open!");
  }

  std::string tmp;

  std::getline(istr, tmp);

  if (tmp != "ColorLUT") {
    throw std::runtime_error("expected ColorLUT");
  }

  const int numbers[3] = { ybits, cbits, numcolors };

  for (int i=0; i<3; ++i) {
    std::getline(istr, tmp);
    if (tmp != int2str(numbers[i])) {
      throw std::runtime_error("Loaded values did not match given paramenters");
    }
  }

  for (size_t i=0; i<numcolors; ++i) {
    std::getline(istr, colornames[i]);
  }

  istr.read( (char*)(&lutdata[0]),
	     sizeof(colorflags)*lutdata.size() );

  
  if (!istr) {
    throw std::runtime_error("error reading");
  }

  istr.get();

  if (!istr.eof()) {
    throw std::runtime_error("garbage at end of file!?@!?");
  }


}


static void setupMat(cv::Mat& m, int rows, int cols, int type) {
  if (m.rows != rows ||
      m.cols != cols ||
      m.type() != type) {
    m = cv::Mat(rows, cols, type);
  }
}

/*Given an image, construct a representation of it out of the known colorflags*/
void ColorLUT::getImageColors(const cv::Mat& image,  cv::Mat& cf) const {

  assert( image.type() == CV_8UC3 );

  // we want colorflags to be of type CV_8UC1
  // and same size as image

  setupMat(cf, image.rows, image.cols, CV_8UC1);

  for (int i=0; i<image.rows; ++i) {
    for (int j=0; j<image.cols; ++j) {
      pixel p = image.at<pixel>(i,j);
      cf.at<colorflags>(i,j) = getColors(p);
    }
  }

}

/*Given an image, construct a mask that is positive for the given color*/
void ColorLUT::getImageColor(const cv::Mat& image, 
			     size_t cidx, 
			     cv::Mat& mask) const {

  assert( image.type() == CV_8UC3 );

  // we want colorflags to be of type CV_8UC1
  // and same size as image

  setupMat(mask, image.rows, image.cols, CV_8UC1);

  colorflags flags = (1 << cidx);

  for (int i=0; i<image.rows; ++i) {
    for (int j=0; j<image.cols; ++j) {
      pixel p = image.at<pixel>(i,j);
      colorflags f = getColors(p);
      /*If the flag at pixel p is the same as the flag for cidx, set the mask
      to white at the pixel, otherwise, set to black*/
      mask.at<unsigned char>(i,j) = (f & flags) ? 255 : 0;
    }
  }

}

void ColorLUT::colorFlagsToMask(const cv::Mat& cf, 
				size_t cidx,
				cv::Mat& mask) const {

  assert(cf.type() == CV_8UC1);
  setupMat(mask, cf.rows, cf.cols, CV_8UC1);

  colorflags flags = (1 << cidx);

  for (int i=0; i<cf.rows; ++i) {
    for (int j=0; j<cf.cols; ++j) {
      colorflags f = cf.at<colorflags>(i,j);
      mask.at<unsigned char>(i,j) = (f & flags) ? 255 : 0;
    }
  }

}


void ColorLUT::getContours(const cv::Mat& mask,
			   PointVecs& regions) const {

  cv::Mat work = mask.clone();

  cv::findContours(work, regions,
		   CV_RETR_CCOMP,
		   CV_CHAIN_APPROX_SIMPLE);

}

bool cmpRInfo(const ColorLUT::RegionInfo& r1,
	      const ColorLUT::RegionInfo& r2) {
  return r1.area > r2.area;
}

void ColorLUT::getRegionInfo(const cv::Mat& mask, 
			     RegionInfoVec& info) const {

  PointVecs regions;

  getContours(mask, regions);

  info.clear();

  for (size_t i=0; i<regions.size(); ++i) {
    info.push_back(RegionInfo());
    contourToRegionInfo(regions[i], info.back());
  }

  std::sort(info.begin(), info.end(), cmpRInfo);

}

void ColorLUT::contourToRegionInfo(const PointVec& region,
				   RegionInfo& info) const {

  cv::Moments m = cv::moments(cv::Mat(region));

  info.area = m.m00;

  if (!info.area) {

    info.mean = info.b1 = info.b2 = cv::Point2d(0,0);

  } else {

    info.mean.x = m.m10 / m.m00;
    info.mean.y = m.m01 / m.m00;

    cv::Mat_<float> A(2,2);

    A(0,0) = m.mu20 / m.m00;  A(0,1) = m.mu11 / m.m00;
    A(1,0) = m.mu11 / m.m00;  A(1,1) = m.mu02 / m.m00;

    cv::SVD svd(A);

    cv::Mat_<float> U = svd.u;
    cv::Mat_<float> W = svd.w;

    info.b1 = sqrt(W(0)) * cv::Point2d(U(0,0), U(1,0));
    info.b2 = sqrt(W(1)) * cv::Point2d(U(0,1), U(1,1));
    

  }

}
