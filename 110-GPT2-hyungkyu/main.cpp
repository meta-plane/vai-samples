void tokenizerTest();
void dataLoaderTest();
void embeddingNodeTest();
void attentionNodeTest();
void transformerNodeTest();
void gpt2Test();
void generationTest();

int main()
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

    // Run text generation tests FIRST (before GPU resources are exhausted)
    generationTest();

    // Run GPT-2 model tests (may fail on full-size model due to GPU limits)
    gpt2Test();

    return 0;
}

