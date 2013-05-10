#ifndef DEVICE_H_
#define DEVICE_H_

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

extern ErrorStatus Device_measure(Uint8 , Uint32*);


#endif /* DEVICE_H_ */
