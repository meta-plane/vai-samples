# PyTorch Convention Refactor Progress

## Phase 1: Shader & Node Changes
- [x] BatchNorm1D
- [x] MaxPool1D
- [x] Broadcast
- [x] Concat
- [x] GEMM + PointWiseMLP
- [x] MatMul
- [x] Slice
- [x] AddIdentity (no change needed - square matrix)
- [x] FullyConnected

## Phase 2: Test References (14 tests)
- [x] test/batchnorm
- [x] test/mlp
- [x] test/fc
- [ ] test/fcbn
- [x] test/maxpool
- [ ] test/matmul
- [ ] test/add_identity
- [ ] test/mlpseq
- [ ] test/fcseq
- [ ] test/fcbn_seq
- [ ] test/tnet
- [ ] test/encoder
- [ ] test/segment
- [ ] test/validation

## Phase 3: Weight Loading
- [ ] weights.cpp transpose 제거
- [ ] SafeTensors 로딩 검증

## Phase 4: Verification
- [ ] All unit tests pass
- [ ] Integration test pass
- [ ] Performance benchmark

