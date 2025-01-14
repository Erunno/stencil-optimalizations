#ifndef LEXICON_HPP
#define LEXICON_HPP

#include "patterns.hpp"
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace lexicon {
class ExprUtils {
  public:
    static std::size_t skip_ws(const std::string& expression, std::size_t current_index) {
        while (current_index < expression.size() && is_ws(expression[current_index])) {
            current_index++;
        }
        return current_index;
    }

    static bool is_ws(char c) {
        return std::isspace(c) || c == ';';
    }

    std::string remove_ws(const std::string& expression) {
        std::string result;
        for (auto&& c : expression) {
            if (!is_ws(c)) {
                result.push_back(c);
            }
        }
        return result;
    }
};

class PatternExpresionRecord {
  public:
    PatternExpresionRecord(const std::string& name, const std::string& x_str, const std::string& y_str)
        : _name(name), _x_str(x_str), _y_str(y_str) {
    }

    static std::tuple<std::size_t, PatternExpresionRecord> from(const std::string& expression,
                                                                std::size_t current_index) {

        std::string name;
        std::string x_str;
        std::string y_str;

        current_index = load_name(&name, expression, current_index);
        current_index = load_coord(&x_str, expression, current_index + 1);
        current_index = load_coord(&y_str, expression, current_index + 1);

        return {current_index, PatternExpresionRecord(name, x_str, y_str)};
    }

    std::string name() const {
        return _name;
    }

    std::size_t x() const {
        return std::stoi(_x_str);
    }

    std::size_t y() const {
        return std::stoi(_y_str);
    }

  private:
    std::string _name;
    std::string _x_str;
    std::string _y_str;

    static std::size_t load_name(std::string* name, const std::string& expression, std::size_t current_index) {
        while (current_index < expression.size() && expression[current_index] != '[' &&
               !ExprUtils::is_ws(expression[current_index])) {
            name->push_back(expression[current_index]);
            current_index++;
        }
        return current_index;
    }

    static std::size_t load_coord(std::string* coord, const std::string& expression, std::size_t current_index) {
        while (current_index < expression.size() && expression[current_index] != ',' &&
               expression[current_index] != ']' && !ExprUtils::is_ws(expression[current_index])) {

            coord->push_back(expression[current_index]);
            current_index++;
        }
        return current_index;
    }
};

class PatterExpression {

  public:
    PatterExpression(const std::vector<PatternExpresionRecord>& records) : _records(records) {
    }

    static PatterExpression from(const std::string& expression) {
        std::vector<PatternExpresionRecord> records;

        auto trimmed_expression = ExprUtils().remove_ws(expression);

        std::size_t current_index = 0;
        while (current_index < trimmed_expression.size()) {
            auto [new_index, record] = PatternExpresionRecord::from(trimmed_expression, current_index);
            records.push_back(record);
            current_index = new_index + 1;
        }

        return PatterExpression(records);
    }

    std::vector<PatternExpresionRecord> records() const {
        return _records;
    }

  private:
    std::vector<PatternExpresionRecord> _records;
};

class Lexicon {
  public:
    Lexicon() {
        _patterns = PatternDict().all_patterns();
    }

    template <typename Grid>
    void insert_patters(Grid& grid, const std::string& pattern_expression) {
        auto expression = PatterExpression::from(pattern_expression);

        for (auto&& record : expression.records()) {
            insert(grid, record);
        }
    }

  private:
    template <typename Grid>
    void insert(Grid& grid, const PatternExpresionRecord& record) {
        auto pattern = get_pattern(record.name());

        auto x_offset = record.x();
        auto y_offset = record.y();

        for (std::size_t y = 0; y < pattern->size(); y++) {
            auto&& row = (*pattern)[y];

            for (std::size_t x = 0; x < row.size(); x++) {
                auto value = static_cast<typename Grid::element_t>(row[x]);
                grid[x_offset + x][y_offset + y] = value;
            }
        }
    }

    Pattern* get_pattern(const std::string& name) {
        if (_patterns.find(name) == _patterns.end()) {
            return &_empty_pattern;
        }
        return &_patterns[name];
    }

    std::map<std::string, Pattern> _patterns;
    Pattern _empty_pattern = PatternDict::empty_pattern();
};

} // namespace lexicon

#endif // LEXICON_HPP
