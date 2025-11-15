void tokenizerTest();
void dataLoaderTest();
void embeddingNodeTest();

int main()
{
    // Run tokenizer tests
    tokenizerTest();

    // Run dataloader tests
    dataLoaderTest();

    // Run embedding node tests (Vulkan version)
    embeddingNodeTest();

    return 0;
}

