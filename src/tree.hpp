#ifndef TREE_HPP
#define TREE_HPP

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <tuple>

namespace test {

using num = std::int_fast8_t;

struct Env {
    num alive;
    num neighbors;
};

class Node {
public:
    virtual ~Node() = default;
    int virtual eval(Env env) = 0;
    std::string virtual to_string() = 0;
};

using node_ptr = std::unique_ptr<Node>;

template <num value>
class Const : public Node {
public:
    int eval(Env env) override { (void)env; return value; }
    std::string to_string() override { return std::to_string(value); }
};


class AliveVar : public Node {
public:
    int eval(Env env) override { return env.alive; }
    std::string to_string() override { return "alive"; }
};

class NeighborhoodVar : public Node {
public:
    int eval(Env env) override { return env.neighbors; }
    std::string to_string() override { return "neighbors"; }
};

class UnOp : public Node {
public:
    virtual ~UnOp() = default;
    UnOp(Node* child) { this->child = child; }
    virtual node_ptr clone() = 0; 
    void change_child(Node* child) { this->child = child; }
protected:
    Node* child;
};

template <num factor>
class Mul : public UnOp {
public:
    Mul(Node* child) : UnOp(child) {}
    
    node_ptr clone() override { return std::make_unique<Mul<factor>>(child); }
    int eval(Env env) override { return child->eval(env) * factor; }
    std::string to_string() override { return "(" + std::to_string(factor) + " * " + child->to_string() + ")"; }
};

template <num shift>
class RightShift : public UnOp {
public:
    RightShift(Node* child) : UnOp(child) {}

    node_ptr clone() override { return std::make_unique<RightShift<shift>>(child); }
    int eval(Env env) override { return child->eval(env) >> shift; }
    std::string to_string() override { return "(" + child->to_string() + " >> " + std::to_string(shift) + ")"; }
};

class BinOp : public Node {
public:
    virtual ~BinOp() = default;
    BinOp(Node* left, Node* right) { this->left = left; this->right = right; }
    virtual node_ptr clone() = 0;
    void change_child(Node* left, Node* right) { this->left = left; this->right = right; }
protected:
    Node* left;
    Node* right;
};

class Plus : public BinOp {
public:
    Plus(Node* left, Node* right) : BinOp(left, right) {}

    node_ptr clone() override { return std::make_unique<Plus>(left, right); }
    int eval(Env env) override { return left->eval(env) + right->eval(env); }
    std::string to_string() override { return "(" + left->to_string() + " + " + right->to_string() + ")"; }
};

class Minus : public BinOp {
public:
    Minus(Node* left, Node* right) : BinOp(left, right) {}

    node_ptr clone() override { return std::make_unique<Minus>(left, right); }
    int eval(Env env) override { return left->eval(env) - right->eval(env); }
    std::string to_string() override { return "(" + left->to_string() + " - " + right->to_string() + ")"; }
};

class And : public BinOp {
public:
    And(Node* left, Node* right) : BinOp(left, right) {}
    node_ptr clone() override { return std::make_unique<And>(left, right); }
    int eval(Env env) override { return left->eval(env) & right->eval(env); }
    std::string to_string() override { return "(" + left->to_string() + " & " + right->to_string() + ")"; }
};

class Or : public BinOp {
public:
    Or(Node* left, Node* right) : BinOp(left, right) {}
    node_ptr clone() override { return std::make_unique<Or>(left, right); }
    int eval(Env env) override { return left->eval(env) | right->eval(env); }
    std::string to_string() override { return "(" + left->to_string() + " | " + right->to_string() + ")"; }
};

class Xor : public BinOp {
public:
    Xor(Node* left, Node* right) : BinOp(left, right) {}
    node_ptr clone() override { return std::make_unique<Xor>(left, right); }
    int eval(Env env) override { return left->eval(env) ^ right->eval(env); }
    std::string to_string() override { return "(" + left->to_string() + " ^ " + right->to_string() + ")"; }
};

class None : public Node {
public:
    int eval(Env env) override { (void)env; throw std::runtime_error("None node cannot be evaluated"); }
};

class NodeConstructor {
public:
    virtual std::unique_ptr<UnOp> make(Node* child) = 0;
    virtual std::unique_ptr<BinOp> make(Node* left, Node* right) = 0;
};

template <typename node_type>
class BinC : public NodeConstructor {
public:
    std::unique_ptr<UnOp> make(Node* child) override { (void)child; return nullptr; }
    std::unique_ptr<BinOp> make(Node* left, Node* right) override { return std::make_unique<node_type>(left, right); }
};

template <typename node_type>
class UnC : public NodeConstructor {
public:
    std::unique_ptr<UnOp> make(Node* child) override { return std::make_unique<node_type>(child); }
    std::unique_ptr<BinOp> make(Node* left, Node* right) override { (void)left; (void)right; return nullptr; }
};

struct GlobalCache {

    GlobalCache() {
        std::vector<node_ptr> nullary_trees;
        nullary_trees.push_back(std::make_unique<AliveVar>());
        nullary_trees.push_back(std::make_unique<NeighborhoodVar>());
        nullary_trees.push_back(std::make_unique<Const<0>>());
        nullary_trees.push_back(std::make_unique<Const<1>>());
        nullary_trees.push_back(std::make_unique<Const<2>>());
        nullary_trees.push_back(std::make_unique<Const<3>>());
        nullary_trees.push_back(std::make_unique<Const<4>>());
        nullary_trees.push_back(std::make_unique<Const<5>>());
        nullary_trees.push_back(std::make_unique<Const<6>>());
        nullary_trees.push_back(std::make_unique<Const<7>>());
        nullary_trees.push_back(std::make_unique<Const<8>>());
        nullary_trees.push_back(std::make_unique<Const<9>>());
        nullary_trees.push_back(std::make_unique<Const<10>>());
        nullary_trees.push_back(std::make_unique<Const<11>>());
        nullary_trees.push_back(std::make_unique<Const<12>>());
        nullary_trees.push_back(std::make_unique<Const<13>>());
        nullary_trees.push_back(std::make_unique<Const<14>>());
        nullary_trees.push_back(std::make_unique<Const<15>>());
        precomputed_trees.push_back(std::move(nullary_trees));

        unary_ctors.push_back(std::make_unique<UnC<Mul<2>>>());
        unary_ctors.push_back(std::make_unique<UnC<Mul<3>>>());
        unary_ctors.push_back(std::make_unique<UnC<Mul<4>>>());
        unary_ctors.push_back(std::make_unique<UnC<Mul<5>>>());
        unary_ctors.push_back(std::make_unique<UnC<Mul<6>>>());
        unary_ctors.push_back(std::make_unique<UnC<Mul<7>>>());
        unary_ctors.push_back(std::make_unique<UnC<Mul<8>>>());
        unary_ctors.push_back(std::make_unique<UnC<Mul<9>>>());
        unary_ctors.push_back(std::make_unique<UnC<Mul<10>>>());
        unary_ctors.push_back(std::make_unique<UnC<Mul<11>>>());
        unary_ctors.push_back(std::make_unique<UnC<Mul<12>>>());
        unary_ctors.push_back(std::make_unique<UnC<Mul<13>>>());
        unary_ctors.push_back(std::make_unique<UnC<Mul<14>>>());
        unary_ctors.push_back(std::make_unique<UnC<Mul<15>>>());
        unary_ctors.push_back(std::make_unique<UnC<RightShift<1>>>());
        unary_ctors.push_back(std::make_unique<UnC<RightShift<2>>>());
        unary_ctors.push_back(std::make_unique<UnC<RightShift<3>>>());

        binary_ctors.push_back(std::make_unique<BinC<Plus>>());
        binary_ctors.push_back(std::make_unique<BinC<Minus>>());
        binary_ctors.push_back(std::make_unique<BinC<And>>());
        binary_ctors.push_back(std::make_unique<BinC<Or>>());
        binary_ctors.push_back(std::make_unique<BinC<Xor>>());
    }    

    std::vector<std::unique_ptr<NodeConstructor>> unary_ctors;

    std::vector<std::unique_ptr<NodeConstructor>> binary_ctors;

    std::vector<std::vector<node_ptr>> precomputed_trees;
    std::vector<node_ptr> currently_computed_trees;

    bool is_precomputed(num order) {
        return static_cast<std::size_t> (order) < precomputed_trees.size();
    }

    std::tuple<node_ptr*, std::size_t> get_precomputed(num order) {
        return { precomputed_trees[order].data(), precomputed_trees[order].size() };
    }

    void finalize_computed_trees() {
        precomputed_trees.push_back(std::move(currently_computed_trees));
        currently_computed_trees.clear();
    }
};


class Generator {
public:
    Generator(GlobalCache& cache, std::size_t order) 
        : cache(cache), order(order), index(0), cached_trees(nullptr), cached_size(0),
          next_generator(nullptr), left_generator(nullptr), right_generator(nullptr),
          current_unary(nullptr), current_binary(nullptr), current_left_subtree(nullptr) {}

    Node* next() {
        if (order < 0) {
            std::cout << "order < 0" << std::endl;
            return nullptr;
        }

        if (cache.is_precomputed(order)) {
            std::cout << "precomputed" << std::endl;
            return load_trees_from_cache();
        }

        if (index < cache.unary_ctors.size()) {
            std::cout << "unary" << std::endl;
            return next_unary();
        }

        if (index < cache.unary_ctors.size() + cache.binary_ctors.size()) {
            std::cout << "binary" << std::endl;
            return next_binary();
        }

        return nullptr;
    }

    node_ptr get_copy_of_current() {
        return current_unary != nullptr ? current_unary->clone() : current_binary->clone();
    }

    void reset() {
        index = 0;
        current_unary = nullptr;
        current_binary = nullptr;
        
        if (next_generator != nullptr) {
            next_generator->reset();
        }
        if (left_generator != nullptr) {
            left_generator->reset();
        }
        if (right_generator != nullptr) {
            right_generator->reset();
        }
    }

private:
    GlobalCache& cache;
    num order;
    std::size_t index;

    node_ptr* cached_trees;
    std::size_t cached_size;

    std::unique_ptr<Generator> next_generator;
    std::unique_ptr<Generator> left_generator;
    std::unique_ptr<Generator> right_generator;

    std::unique_ptr<UnOp> current_unary;
    std::unique_ptr<BinOp> current_binary;

    Node* current_left_subtree;

    Node* load_trees_from_cache() {
        if (cached_trees == nullptr) {
            auto [trees, size] = cache.get_precomputed(order);
            cached_trees = trees;
            cached_size = size;

        }

        std::cout << "cached_size: " << cached_size << std::endl;
        std::cout << "index: " << index << std::endl;
        
        if (index < cached_size) {
            return cached_trees[index++].get();
        }
        
        return nullptr;
    }

    Node* next_unary() {
        if (next_generator == nullptr) {
            next_generator = std::make_unique<Generator>(cache, order - 1);
        }

        while (true) {
            auto next_subtree = next_generator->next();

            if (next_subtree != nullptr) {
                
                if (current_unary == nullptr) {
                    current_unary = cache.unary_ctors[index]->make(next_subtree);
                }
                else {
                    current_unary->change_child(next_subtree);
                }

                return current_unary.get();
            }

            index++;

            if (index < cache.unary_ctors.size()) {
                current_unary = nullptr;
                next_generator = nullptr;
                return next_binary();
            }

            next_generator->reset();
            current_unary = nullptr;
        }
    }

    Node* next_binary() {
        if (order < 2) {
            return nullptr;
        }

        if (left_generator == nullptr) {
            renew_left_right_generators();
        }

        while (true) {
            auto next_right_subtree = right_generator->next();

            if (next_right_subtree != nullptr) {
                if (current_binary == nullptr) {
                    current_binary = cache.binary_ctors[index - cache.unary_ctors.size()]->make(current_left_subtree, next_right_subtree);
                }
                else {
                    current_binary->change_child(current_left_subtree, next_right_subtree);
                }

                return current_binary.get();
            }

            current_left_subtree = left_generator->next();

            if (current_left_subtree != nullptr) {
                right_generator->reset();
                continue;
            }

            if (can_move_to_next_generators()) {
                move_to_next_generators();
                continue;
            }

            index++;
            current_binary = nullptr;
            renew_left_right_generators();

            if (index >= cache.unary_ctors.size() + cache.binary_ctors.size()) {
                return nullptr;
            }
        }
    }

    void renew_left_right_generators() {
        left_generator = std::make_unique<Generator>(cache, order - 1);
        right_generator = std::make_unique<Generator>(cache, 0);

        current_left_subtree = left_generator->next();
    }

    void move_to_next_generators() {
        left_generator = std::make_unique<Generator>(cache, left_generator->order - 1);
        right_generator = std::make_unique<Generator>(cache, right_generator->order + 1);

        current_left_subtree = left_generator->next();
    }

    bool can_move_to_next_generators() {
        return left_generator->order > 0;
    }

};

template <typename whatever>
void run_search() {

    GlobalCache cache;

    Generator generator(cache, 0);

    auto tree = generator.next();
    std::cout << "tree: " << std::endl;

    std::cout << tree->to_string() << std::endl;
}


}

#endif // TREE_HPP