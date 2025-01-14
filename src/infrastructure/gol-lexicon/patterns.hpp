#include <map>
#include <string>
#include <vector>

namespace lexicon {

using CellState = bool;
using Pattern = std::vector<std::vector<CellState>>;

class PatternDict {
  public:
    static Pattern empty_pattern() {
        return {{}};
    }

    std::map<std::string, Pattern> all_patterns();
};

} // namespace lexicon
