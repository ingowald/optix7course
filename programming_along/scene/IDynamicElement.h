
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

	bool GetDynamicEnabled() const
	{
		return DynamicEnabled;
	}

	virtual void SetDynamicEnabled(const bool& enabled)
	{
		DynamicEnabled = enabled;
	}

protected:
	bool DirtyBit = false;
	bool DynamicEnabled = true;
};