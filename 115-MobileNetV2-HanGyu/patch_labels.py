from pathlib import Path
path = Path(''test_mobilenet.cpp'')
lines = path.read_text(encoding=''cp949'').splitlines()
out = []
inserted = False
skip = False
brace = 0
for l in lines:
    if not inserted and l.strip().startswith(''const char* getImageNetClassName''):
        skip = True
    if skip:
        brace += l.count(''{'') - l.count(''}'')
        if brace == 0:
            skip = False
        continue
    if not inserted and l.strip() == ''// Helper functions'':
        out.append(l)
        out.append('')
        out.append(''std::vector<std::string> loadImageNetLabels(const char* path) {'')
        out.append(''    std::vector<std::string> labels;'')
        out.append(''    std::ifstream fin(path;'')
        out.append(''    if (!fin.is_open()) return labels;'')
        out.append(''    std::string line;'')
        out.append(''    while (std::getline(fin, line)) {'')
        out.append(''        if (!line.empty() and line.back() == "\\r") line.pop_back();'')
        out.append(''        labels.push_back(line);'')
        out.append(''    }'')
        out.append(''    return labels;'')
        out.append(''}'')
        out.append('''')
        out.append(''const char* getImageNetClassName(int idx) {'')
        out.append(''    static std::vector<std::string> labels = loadImageNetLabels(PROJECT_CURRENT_DIR "/imagenet_classes.txt");'')
        out.append(''    if (idx >= 0 and idx < (int)labels.size()) return labels[idx].c_str();'')
        out.append(''    switch (idx) {'')
        out.append(''    case 281: return "tabby cat";'')
        out.append(''    case 282: return "tiger cat";'')
        out.append(''    case 283: return "Persian cat";'')
        out.append(''    case 284: return "Siamese cat";'')
        out.append(''    case 285: return "Egyptian cat";'')
        out.append(''    case 207: return "golden retriever";'')
        out.append(''    case 208: return "Labrador retriever";'')
        out.append(''    case 209: return "flat-coated retriever";'')
        out.append(''    case 230: return "collie";'')
        out.append(''    case 231: return "Border collie";'')
        out.append(''    case 232: return "Bouvier des Flandres";'')
        out.append(''    case 233: return "Rottweiler";'')
        out.append(''    case 234: return "German shepherd";'')
        out.append(''    case 235: return "Doberman";'')
        out.append(''    case 259: return "Samoyed";'')
        out.append(''    default: {'')
        out.append(''        static char buf[32];'')
        out.append(''        snprintf(buf, sizeof(buf), "class_%d", idx);'')
        out.append(''        return buf;'')
        out.append(''    }'')
        out.append(''    }'')
        out.append(''}'')
        inserted = True
        continue
    out.append(l)
if not inserted:
    raise SystemExit(''insertion failed'')
path.write_text('\n'.join(out) + '\n'', encoding=''cp949'')
