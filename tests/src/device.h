#ifndef DEVICE_H_
#define DEVICE_H_

#include <stdio.h>
#include <stdint.h>
#include "reftypes.h"

#define DEVICE_MAX_NUM 20

typedef ErrorStatus (*DeviceMeasureFunc)(Uint32*);

typedef struct DeviceDescriptorTag
{
	Uint8 deviceNumber;
	DeviceMeasureFunc deviceMeasurement;

} DeviceDescriptor;

typedef struct DeviceParamTag
{
	DeviceDescriptor* deviceDescriptor;

} DeviceParam;

extern ErrorStatus Device_Init(DeviceParam*);

extern ErrorStatus Device_measure1(Uint32*);
extern ErrorStatus Device_measure2(Uint8 , Uint32*);

// typedef double (*TwoNumsOp)(TwoNums *);

typedef struct TwoNums
{
  double a;
  double b;
  //TwoNumsOp op;
  //double (*op)(TwoNumsTag *);
  double (*op)(double, double);
} TwoNums;

//int64_t afunc(int64_t, float);


#endif /* DEVICE_H_ */
