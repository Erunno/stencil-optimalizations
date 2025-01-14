#ifndef INFRASTRUCTURE_ALGORITHM_REPOSITORY_HPP
#define INFRASTRUCTURE_ALGORITHM_REPOSITORY_HPP

#include "algorithm.hpp"
#include <memory>
#include <string>
#include <unordered_map>

namespace infrastructure {

template <int Dims, typename ElementType>
class AlgorithmRepository {
  public:
    using AlgType = Algorithm<Dims, ElementType>;

    void register_algorithm(const std::string& algorithm_name, std::unique_ptr<AlgType> algorithm) {
        _algorithms.insert_or_assign(algorithm_name, std::move(algorithm));
    }

    template <typename Alg>
    void register_algorithm(const std::string& algorithm_name) {
        _algorithms.insert_or_assign(algorithm_name, std::make_unique<Alg>());
    }

    void register_loader(const std::string& loader_name, std::unique_ptr<AlgType> loader) {
        _algorithms.insert_or_assign(loader_name, std::move(loader));
    }

    AlgType* fetch_algorithm(const std::string& algorithm_name) {
        const auto it = _algorithms.find(algorithm_name);
        if (it != _algorithms.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    const AlgType* fetch_algorithm(const std::string& algorithm_name) const {
        const auto it = _algorithms.find(algorithm_name);
        if (it != _algorithms.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    bool has_algorithm(const std::string& algorithm_name) const {
        return _algorithms.contains(algorithm_name);
    }

  private:
    std::unordered_map<std::string, std::unique_ptr<AlgType>> _algorithms;
};

template <int Dims, typename ElementType>
class AlgRepoParams {
  public:
    static constexpr int DIMS = Dims;
    using ElementT = ElementType;
};

template <typename... RepoDescriptors>
class AlgorithmReposCollection {};

template <typename RepoDescriptor, typename... RepoDescriptors>
class AlgorithmReposCollection<RepoDescriptor, RepoDescriptors...> {
  private:
    AlgorithmRepository<RepoDescriptor::DIMS, typename RepoDescriptor::ElementT> _repo;
    AlgorithmReposCollection<RepoDescriptors...> _next;

  public:
    template <int Dims, typename ElementType>
    AlgorithmRepository<Dims, ElementType>* get_repository() {
        if constexpr (Dims == RepoDescriptor::DIMS &&
                      std::is_same<ElementType, typename RepoDescriptor::ElementT>::value) {
            return &_repo;
        }
        else {
            return _next.template get_repository<Dims, ElementType>();
        }
    }

    template <typename Func>
    void for_each(Func&& func) {
        func(_repo);
        _next.for_each(std::forward<Func>(func));
    }
};

template <>
class AlgorithmReposCollection<> {
  public:
    template <int Dims, typename ElementType>
    AlgorithmRepository<Dims, ElementType>* get_repository() {
        return nullptr;
    }

    template <typename Func>
    void for_each(Func&& func) {
        (void)func;
    }
};

} // namespace infrastructure

#endif // INFRASTRUCTURE_ALGORITHM_REPOSITORY_HPP
