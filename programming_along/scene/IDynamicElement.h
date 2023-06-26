
#pragma once

class IDynamicElement
{
public:
	bool IsMarkedDirty()
	{
		return DirtyBit;
	}

protected:
	bool DirtyBit = false;
};