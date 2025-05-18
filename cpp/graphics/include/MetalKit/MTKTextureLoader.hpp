/*
 *
 * Copyright 2020-2021 Apple Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// MetalKit/MTKTextureLoader.hpp
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#include "MetalKitPrivate.hpp"

#include <AppKit/AppKit.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <CoreGraphics/CGColorSpace.h>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace MTK
{
	using TextureLoaderOption = class String*;

	//_NS_CONST(TextureLoaderOption, TextureLoaderOptionAllocatedMipmaps);
	_NS_CONST(TextureLoaderOption, TextureLoaderOptionGenerateMipmaps);
	_NS_CONST(TextureLoaderOption, TextureLoaderOptionTextureCPUCacheMode);
	_NS_CONST(TextureLoaderOption, TextureLoaderOptionTextureStorageMode);
	_NS_CONST(TextureLoaderOption, TextureLoaderOptionTextureUsage);
	_NS_CONST(TextureLoaderOption, TextureLoaderOptionOrigin);
	_NS_CONST(TextureLoaderOption, TextureLoaderOptionSRGB);
	_NS_CONST(TextureLoaderOption, TextureLoaderOptionLoadAsArray);
	_NS_CONST(TextureLoaderOption, TextureLoaderOptionCubeLayout);

	class TextureLoader : public NS::Referencing<MTK::TextureLoader> {
	public:
		static TextureLoader* alloc();

		TextureLoader* init(MTL::Device* device);

		MTL::Texture* newTexture(NS::Data* data, NS::Dictionary* options, NS::Error** error);

	};

	_NS_INLINE TextureLoader* MTK::TextureLoader::alloc()
	{
		return NS::Object::alloc<TextureLoader>(_MTK_PRIVATE_CLS(MTKTextureLoader));
	}

	_NS_INLINE TextureLoader* MTK::TextureLoader::init(MTL::Device* device)
	{
		return NS::Object::sendMessage<TextureLoader*>(this, _MTK_PRIVATE_SEL(initWithDevice_), device);
	}

	_NS_INLINE MTL::Texture* MTK::TextureLoader::newTexture(NS::Data* data, NS::Dictionary* options, NS::Error** error)
	{
		return NS::Object::sendMessage<MTL::Texture*>(this,
			_MTK_PRIVATE_SEL(newTextureWithData_options_error_),
			data,
			options,
			error);
	}
}

//_MTK_PRIVATE_DEF_CONST(MTK::TextureLoaderOption, TextureLoaderOptionAllocatedMipmaps);
_MTK_PRIVATE_DEF_CONST(MTK::TextureLoaderOption, TextureLoaderOptionGenerateMipmaps);
_MTK_PRIVATE_DEF_CONST(MTK::TextureLoaderOption, TextureLoaderOptionTextureCPUCacheMode);
_MTK_PRIVATE_DEF_CONST(MTK::TextureLoaderOption, TextureLoaderOptionTextureStorageMode);
_MTK_PRIVATE_DEF_CONST(MTK::TextureLoaderOption, TextureLoaderOptionTextureUsage);
_MTK_PRIVATE_DEF_CONST(MTK::TextureLoaderOption, TextureLoaderOptionOrigin);
_MTK_PRIVATE_DEF_CONST(MTK::TextureLoaderOption, TextureLoaderOptionSRGB);
_MTK_PRIVATE_DEF_CONST(MTK::TextureLoaderOption, TextureLoaderOptionLoadAsArray);
_MTK_PRIVATE_DEF_CONST(MTK::TextureLoaderOption, TextureLoaderOptionCubeLayout);
