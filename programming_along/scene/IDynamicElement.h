
#pragma once

class IDynamicElement
{
public:
	bool IsMarkedDirty()
	{
		return DirtyBit;
	}

	void SetDirtyBitConsumed()
	{
		DirtyBit = false;
	}

protected:
	bool DirtyBit = false;
};