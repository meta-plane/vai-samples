<!-- ee764c2e-763e-42da-8194-6cce6810a4e1 acf45bba-6ba2-4f37-b8fc-cd695a38619b -->
# MobileNetV2 프로젝트 구현 계획

## 현재 상태 분석

- **기본 구조**: Tensor, Node, NeuralNet의 스켈레톤만 존재
- **레퍼런스**: 완전한 그래프 기반 실행 엔진 (NodeSlot/Edge, Topological sort, Vulkan 통합)

## 구현 단계

### Phase 1: Tensor 시스템 확장

**목표**: Host/Device 데이터 관리 및 BufferPool 구현

**파일**: `core/tensor.h`, `core/tensor.cpp`

- TensorData 클래스 추가 (shared_ptr 기반)
- Host data와 Device data (Vulkan buffer) 분리 관리
- BufferPool 싱글톤 구현 (메모리 재사용)
- `hasHostData()`, `hasDeviceData()`, `bindBuffer()` 등 메서드 추가
- `reshape()`, `permute()` 등 유틸리티 메서드

### Phase 2: NodeSlot과 Edge 시스템

**목표**: 노드 간 데이터 플로우 연결 메커니즘

**파일**: `core/neural_net.h`, `core/neural_net.cpp`

- NodeSlot 클래스 (input/output/internal 타입)
- Edge 클래스 (노드 간 연결)
- Node 클래스에 slots 맵 추가
- `operator[]` 오버로딩으로 slot 접근

### Phase 3: NeuralNet 그래프 실행 엔진

**목표**: Topological sort 기반 실행 및 메모리 관리

**파일**: `core/neural_net.h`, `core/neural_net.cpp`

- InputNode, OutputNode 클래스
- Topological sort 구현 (Kahn's algorithm)
- `prepare()`: 텐서 shape 검증 및 할당
- `run()`: 노드 실행 순서 결정 및 CommandBuffer 기록
- Upload buffer 관리 (Host → Device 전송)
- NodeFlow 연산자 오버로딩 (`operator-`, `operator/`)

### Phase 4: VulkanApp 기본 구조

**목표**: Vulkan 래퍼 클래스 기본 구현

**파일**: `core/vulkan_app.h`, `core/vulkan_app.cpp`

- Device, Buffer, CommandBuffer 등 기본 래퍼 클래스
- Compute queue 설정
- 기본 초기화/정리 로직

### Phase 5: 기본 Node 구현

**목표**: Input/Output 노드 및 간단한 연산 노드

**파일**: `nodes/base_node.h`, `nodes/base_node.cpp`

- BaseNode를 Node 상속으로 변경
- InputNode, OutputNode 구현
- FlattenNode 구현 (copy shader)

### Phase 6: 핵심 연산 Node 구현

**목표**: MobileNetV2에 필요한 연산 노드들

**파일**: `nodes/` (새 파일들)

- **ReluNode**: ReLU 활성화 함수 (compute shader)
- **MaxPoolingNode**: Max pooling (compute shader)
- **ConvolutionNode**: Convolution (im2col + GEMM)
- **FullyConnectedNode**: Fully connected layer (GEMM)

각 노드는:

- `prepare()`: 출력 shape 계산 및 descriptor set 준비
- `run(CommandBuffer)`: compute shader 실행

### Phase 7: MobileNetV2 모델 구성

**목표**: MobileNetV2 아키텍처 구현

**파일**: `model/mobilenet_v2.h`, `model/mobilenet_v2.cpp`

- MobileNetV2 클래스가 NeuralNet 상속
- Inverted residual blocks 구성
- `initialize()`: 네트워크 구조 생성
- `loadWeights()`: JSON에서 가중치 로딩

### Phase 8: 유틸리티 및 데이터 로딩

**목표**: 가중치 로딩 및 이미지 처리

**파일**: `utils/json_parser.*`, `dataloader/image_loader.*`

- JsonParser: 가중치 JSON 파싱
- ImageLoader: 이미지 로딩 및 전처리

### Phase 9: 통합 및 테스트

**목표**: End-to-end 실행

**파일**: `main.cpp`

- 전체 파이프라인 통합
- 이미지 입력 → 추론 → 결과 출력

## 구현 우선순위

**1단계 (핵심 인프라)**: Phase 1-3

- Tensor 확장
- NodeSlot/Edge 시스템
- NeuralNet 실행 엔진

**2단계 (Vulkan 통합)**: Phase 4-5

- VulkanApp 기본 구조
- 기본 노드 구현

**3단계 (연산 노드)**: Phase 6

- Conv, ReLU, Pooling, FC 노드

**4단계 (모델 및 통합)**: Phase 7-9

- MobileNetV2 모델
- 유틸리티
- 통합 테스트

## 주요 설계 결정사항

1. **메모리 관리**: BufferPool을 통한 버퍼 재사용으로 메모리 효율성 향상
2. **실행 모델**: Topological sort로 의존성 기반 실행 순서 결정
3. **데이터 플로우**: NodeSlot과 Edge로 명시적 연결 관리
4. **Vulkan 통합**: Compute shader를 통한 GPU 가속

### To-dos

- [ ] Tensor 시스템 확장: TensorData, BufferPool, Host/Device 데이터 관리
- [ ] NodeSlot과 Edge 시스템 구현: 노드 간 연결 메커니즘
- [ ] NeuralNet 그래프 실행 엔진: Topological sort, prepare(), run()
- [ ] VulkanApp 기본 구조: Device, Buffer, CommandBuffer 래퍼
- [ ] 기본 Node 구현: InputNode, OutputNode, FlattenNode
- [ ] 핵심 연산 Node: ReluNode, MaxPoolingNode, ConvolutionNode, FullyConnectedNode
- [ ] MobileNetV2 모델 구성: 아키텍처 및 가중치 로딩
- [ ] 유틸리티 구현: JsonParser, ImageLoader
- [ ] 통합 및 테스트: main.cpp에서 end-to-end 실행