#include "stdlib.h"

#include "device.h"

static DeviceParam deviceParam;

extern ErrorStatus Device_Init(DeviceParam* param)
{
	ErrorStatus retVal;

	if (param != NULL &&
		param->deviceDescriptor != NULL &&
		param->deviceDescriptor->deviceNumber < DEVICE_MAX_NUM &&
		param->deviceDescriptor->deviceMeasurement != 0)
	{
		retVal = ERROR_OK;
		deviceParam = *param;
	}
	else
	{
		retVal = ERROR_FAILED_INIT;

	}
	return retVal;
}

extern ErrorStatus Device_measure(Uint32* aiValue)
{
	return deviceParam.deviceDescriptor->deviceMeasurement(aiValue);
}

extern ErrorStatus Device_measure(Uint8 deviceNumber, Uint32* aiValue)
{
	return deviceParam.deviceDescriptor->deviceMeasurement(aiValue);
}
