#ifndef TENSOR_H
#define TENSOR_H


#include "jsonParser.h"
#include "vulkanApp.h"
#include "error.h"
#include <memory>
#include <vector>
#include <map>
#include <unordered_map>

// Vulkan vx::Buffer들을 재사용하기 위한 버퍼 풀
// 여기서 Buffer는 Vulkan GPU 메모리에 있는 "버퍼 데이터 저장소!"
// C++에는 Tensor에 2종류의 데이터가 존재(1.hostData(CPU), 2. device buffer(GPU VRAM)
// 이 중 GPU 쪽 buffer가 vk::Buffer!!
// 근데 Bufferpool이 왜 필요??
// -> Vulkan에서는 버퍼를 만드는 것이 매우 비쌈(시간,메모리등등) 또, 매 연산마다 새로운 버퍼를 만들고 지우고하면
// 성능저하, GPU 메모리 파편화, 실전 속도 다운 현상이 발생!

// 결국, BufferPool은 "GPU 버퍼 재사용 시스템"
class BufferPool
{
    // vk::Device device;
    std::unordered_map<
        VkBufferUsageFlags,  
        std::multimap<size_t, std::pair<vk::Buffer, VkMemoryPropertyFlags>> > bufferPool;

public:
    static BufferPool& get()
    {
        static BufferPool instance;
        return instance;
    }
    // requestBuffer를 통해 필요한 크기의 버퍼를 요청
    // 재사용 가능한 버퍼가 있으면 곧바로 반환
    // 없으면 새로 만들고 Pool에 집어넣음
    // 사용이 끝난 버퍼는 다시 returnBuffer를 통해 다시 pool에 돌려놓음
    vk::Buffer requestBuffer(
        vk::Device& device,
        uint32_t usageFlags, 
        uint32_t reqMemProps,
        size_t minSize,
        size_t maxSize = size_t(-1)
    )
    {
        auto& subPool = bufferPool[usageFlags];

        for (auto it = subPool.lower_bound(minSize);
             it != subPool.end() && it->first <= maxSize;
             ++it)
        {
            const auto& [buffer, memProps] = it->second;
            if ((memProps & reqMemProps) == reqMemProps)
            {
                vk::Buffer result = std::move(it->second.first);
                subPool.erase(it);
                return result;
            }
        }

        // Create new buffer
        return device.createBuffer({
            .size = minSize,
            .usage = usageFlags,
            .reqMemProps = reqMemProps
        });
    }

    void returnBuffer(vk::Buffer buffer)
    {
        if (buffer) 
        {
            bufferPool[buffer.usage()].emplace(
                buffer.size(), 
                std::make_pair(std::move(buffer), buffer.memoryProperties())
            );
        }
    }
};

// 텐서의 실제 데이터를 들고있는 클래스
// Tensor는 이걸 std::shared_ptr로 감싸서 쓰는 껍데기 역할을 수행

class TensorData
{
    friend class Tensor;
    // Tensor 클래스가 TensorData의 private 멤버에 직접 접근이 가능하도록!
    // std::vector<uint32_t> shape;
    std::vector<float> hostData; 
    vk::Buffer _buffer; 

public:
    ~TensorData()
    {
        BufferPool::get().returnBuffer(_buffer); // temporary code
    }

};


class Tensor 
{
    std::shared_ptr<TensorData> _data;
    std::vector<uint32_t> _shape;
    bool _isConstant = false;

public:
    template <typename... Ts>
    Tensor(Ts... dims) : _data(std::make_shared<TensorData>())
    {
        static_assert((std::conjunction_v<std::is_integral<Ts>...>), "All dims must be integral types");
        _shape = {static_cast<uint32_t>(dims)...};
    }

    Tensor(const std::vector<uint32_t>& shape) : _data(std::make_shared<TensorData>())
    {
        _shape = shape;
    }

    Tensor(const JsonParserRef& json) : _data(std::make_shared<TensorData>())
    {
        set(json.parseNDArray(_shape));
        _isConstant = true;
    }
    
    Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;

    Tensor& set(const std::vector<float>& inData) &
    {
        _ASSERT(!hasDeviceData());  // Force not to set host data if device data is bound
        _ASSERT(numElements() == inData.size());
        _data->hostData = inData;
        return *this;
    }

    Tensor&& set(const std::vector<float>& inData) &&
    {
        _ASSERT(!hasDeviceData());
        _ASSERT(numElements() == inData.size());
        _data->hostData = inData;
        return std::move(*this);
    }

    Tensor& set(std::vector<float>&& inData) &
    {
        _ASSERT(!hasDeviceData());
        _ASSERT(numElements() == inData.size());
        _data->hostData = std::move(inData);
        return *this;
    }

    Tensor&& set(std::vector<float>&& inData) &&
    {
        _ASSERT(!hasDeviceData());
        _ASSERT(numElements() == inData.size());
        _data->hostData = std::move(inData);
        return std::move(*this);
    }

    template <typename... Ts>
    Tensor& reshape(Ts... dims)
    {
        static_assert((std::conjunction_v<std::is_integral<Ts>...>), "All dims must be integral types");
        _ASSERT(numElements() == (static_cast<size_t>(dims) * ...));
        _shape = {static_cast<uint32_t>(dims)...};
        return *this;
    }

    template <typename... Ts>
    Tensor& permute(Ts... Perms);

    const std::vector<uint32_t>& shape() const
    {
        static const std::vector<uint32_t> emptyShape{};
        return _data ? _shape : emptyShape;
    }

    bool isConstant() const
    {
        return _isConstant;
    }

    template <typename... Ts>
    bool isShapeOf(Ts... dims) const
    {
        static_assert((std::conjunction_v<std::is_integral<Ts>...>), "All dims must be integral types");
        if (!_data || _shape.size() != sizeof...(Ts))
            return false;

        const uint32_t dimArray[] = { uint32_t(dims)... };

        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return ((dimArray[Is] == _shape[Is] || dimArray[Is] == uint32_t(-1)) && ...);
        }(std::index_sequence_for<Ts...>{});
    }

    size_t numElements() const
    {
        if (!_data || _shape.empty()) return 0;
        size_t sz = 1; for (uint32_t dim : _shape) sz *= dim;
        return sz;
    }

    bool validShape() const
    {
        if (!_data || _shape.empty()) return false;
        for (uint32_t dim : _shape) if (dim == 0) return false;
        return true;
    }

    bool hasHostData() const
    {
        return _data && !_data->hostData.empty();
    }

    float* hostData()
    {
        if (!_data || _data->hostData.empty()) return nullptr;
        return _data->hostData.data();
    }

    void clearHostData()
    {
        _ASSERT(_data);
        _data->hostData.clear();
    }

    // bool isBufferBound() const
    bool hasDeviceData() const
    {
        return _data && _data->_buffer;
    }

    void bindBuffer(vk::Buffer buf)
    {
        _ASSERT(_data);
        _data->_buffer = buf;
    }

    void unbindBuffer()
    {
        _ASSERT(_data);
        _data->_buffer = vk::Buffer();
    }

    vk::Buffer buffer() const
    {
        _ASSERT(_data && _data->_buffer);
        return _data->_buffer;
    }
};


template <typename... Ts>
inline Tensor& Tensor::permute(Ts... Perms)
{
    static_assert((std::conjunction_v<std::is_integral<Ts>...>), "All dims must be integral types");
    _ASSERT(_data && !_data->hostData.empty());
    
    const uint32_t rank = _shape.size();
    _ASSERT(sizeof...(Perms) == rank);
    _ASSERT(((Perms < rank) && ...)); // Ensure all Perms are valid indices

    const uint32_t permutation[] = { static_cast<uint32_t>(Perms)... };

    for (size_t i = 0; i < rank; ++i)
        for (size_t j = i + 1; j < rank; ++j)
            _ASSERT(permutation[i] != permutation[j]);

    std::vector<uint32_t> dstShape(_shape.size());
    for (size_t i = 0; i < _shape.size(); ++i)
        dstShape[i] = _shape[permutation[i]];

    std::vector<size_t> srcStrides(rank);
    srcStrides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i)
        srcStrides[i] = srcStrides[i + 1] * _shape[i + 1];

    std::vector<size_t> dstStrides(rank);
    dstStrides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i)
        dstStrides[i] = dstStrides[i + 1] * dstShape[i + 1];

    std::vector<float> dstData(_data->hostData.size());
    std::vector<uint32_t> srcIndices(rank);
    std::vector<uint32_t> dstIndices(rank);
    for (size_t linear = 0; linear < _data->hostData.size(); ++linear)
    {
        // linear → multi-dim
        size_t remaining = linear;
        for (uint32_t i = 0; i < rank; ++i)
        {
            srcIndices[i] = remaining / srcStrides[i];
            remaining %= srcStrides[i];
        }

        // permute index
        for (uint32_t i = 0; i < rank; ++i)
            dstIndices[i] = srcIndices[permutation[i]];

        // multi-dim → linear (permuted)
        size_t dstLinear = 0;
        for (uint32_t i = 0; i < rank; ++i)
            dstLinear += dstIndices[i] * dstStrides[i];

        dstData[dstLinear] = _data->hostData[linear];
    }
    
    _shape = std::move(dstShape);
    _data->hostData = std::move(dstData);
    return *this;
}

#endif // TENSOR_H