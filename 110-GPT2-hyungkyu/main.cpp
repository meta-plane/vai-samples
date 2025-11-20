void tokenizerTest();
void dataLoaderTest();
void embeddingNodeTest();
void attentionNodeTest();
void transformerNodeTest();
void testGPT2();
void testGPT2Generation();
void testGPT2Pretrained();

// Run all basic component tests
void runBasicTests()
{
    // Run tokenizer tests
    tokenizerTest();

    // Run dataloader tests
    dataLoaderTest();

    // Run embedding node tests (Vulkan version)
    embeddingNodeTest();

    // Run attention node tests (Multi-Head Attention)
    attentionNodeTest();

    // Run transformer node tests (LayerNorm, GELU, FeedForward)
    transformerNodeTest();

    // Run GPT-2 complete test suite
    testGPT2();

    // Run GPT-2 text generation test (random weights)
    testGPT2Generation();
}

int main()
{
    // Option 1: Run all basic tests (uses GPU memory)
    // runBasicTests();

    // Option 2: Run text generation with random weights (lightweight)
    // testGPT2Generation();

    // Option 3: Run pretrained weights test (requires more GPU memory)
    testGPT2Pretrained();

    // Note: Running both may cause GPU OOM. Choose one or run separately.

    return 0;
}

