#ifndef INFRASTRUCTURE_GRID_HPP
#define INFRASTRUCTURE_GRID_HPP

#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace infrastructure {

template <int Dims, typename ElementType>
class GridTileBase {
  public:
    using size_type = std::size_t;
    using pointer_type = std::add_pointer_t<ElementType>;

    GridTileBase(pointer_type data, size_type* tile_sizes_per_dimensions)
        : _data(data), tile_sizes_per_dimensions(tile_sizes_per_dimensions) {
    }

    pointer_type data() const {
        return _data;
    }

  protected:
    pointer_type _data;
    size_type* tile_sizes_per_dimensions;
};

template <int Dims, typename ElementType>
class GridTile : public GridTileBase<Dims, ElementType> {
  public:
    using pointer_type = typename GridTileBase<Dims, ElementType>::pointer_type;
    using size_type = std::size_t;

    GridTile(pointer_type data, size_type* tile_sizes) : GridTileBase<Dims, ElementType>(data, tile_sizes) {
    }

    using LowerDimTile = GridTile<Dims - 1, ElementType>;
    using LowerDimTileConst = GridTile<Dims - 1, const ElementType>;

    LowerDimTile operator[](int index) {
        auto tile_size = this->tile_sizes_per_dimensions[0];
        return LowerDimTile(this->_data + index * tile_size, this->tile_sizes_per_dimensions + 1);
    }

    LowerDimTileConst operator[](int index) const {
        auto tile_size = this->tile_sizes_per_dimensions[0];
        return LowerDimTileConst(this->_data + index * tile_size, this->tile_sizes_per_dimensions + 1);
    }
};

template <typename ElementType>
class GridTile<1, ElementType> : public GridTileBase<1, ElementType> {
  public:
    using pointer_type = typename GridTileBase<1, ElementType>::pointer_type;
    using size_type = std::size_t;

    GridTile(pointer_type data, size_type* tile_sizes) : GridTileBase<1, ElementType>(data, tile_sizes) {
    }

    auto& operator[](int index) {
        return this->_data[index * this->tile_sizes_per_dimensions[0]];
    }

    const ElementType& operator[](int index) const {
        return this->_data[index * this->tile_sizes_per_dimensions[0]];
    }
};

template <int DIMS, typename ElementType>
class Grid {
  public:
    using size_type = std::size_t;
    using element_t = ElementType;

    Grid() : dimension_sizes(DIMS, 0), tile_sizes_per_dimensions(DIMS, 0) {
    }

    template <typename... Sizes, typename = std::enable_if_t<sizeof...(Sizes) == DIMS>>
    Grid(Sizes... dims) : Grid(std::vector<size_type>{static_cast<size_type>(dims)...}) {
    }

    Grid(const std::vector<size_type>& dimension_sizes) : dimension_sizes(dimension_sizes) {
        if (dimension_sizes.size() != DIMS) {
            throw std::invalid_argument("Dimension sizes must match the number of dimensions");
        }

        size_type total_size = 1;

        for (int i = 0; i < DIMS; i++) {
            tile_sizes_per_dimensions.push_back(total_size);
            total_size *= dimension_sizes[i];
        }

        tile_sizes_per_dimensions.push_back(total_size);

        elements.resize(total_size);
    }

    auto operator[](size_type index) {
        return as_tile()[index];
    }

    auto operator[](size_type index) const {
        return as_const_tile()[index];
    }

    auto as_tile() {
        return GridTile<DIMS, ElementType>(elements.data(), tile_sizes_per_dimensions.data());
    }

    auto as_const_tile() const {
        return GridTile<DIMS, const ElementType>(static_cast<const ElementType*>(elements.data()),
                                                 const_cast<size_type*>(tile_sizes_per_dimensions.data()));
    }

    static constexpr int dimensions() {
        return DIMS;
    }

    ElementType* data() {
        return elements.data();
    }

    const ElementType* data() const {
        return elements.data();
    }

    std::vector<ElementType>* data_as_vector() {
        return &elements;
    }

    size_type size() const {
        return elements.size();
    }

    template <int Dim>
    size_type size_in() const {
        return dimension_sizes[Dim];
    }

    size_type size_in(size_t dim) const {
        return dimension_sizes[dim];
    }

    std::vector<size_type> idx_to_coordinates(size_type idx) const {
        std::vector<size_type> coordinates(DIMS);

        for (int i = DIMS - 1; i >= 0; i--) {
            coordinates[i] = idx % size_in(i);
            idx /= size_in(i);
        }

        return coordinates;
    }

    bool equals(const Grid& other) const {
        for (size_type i = 0; i < elements.size(); i++) {
            if (elements[i] != other.elements[i]) {
                return false;
            }
        }
        return true;
    }

  private:
    std::vector<ElementType> elements;
    std::vector<size_type> dimension_sizes;
    std::vector<size_type> tile_sizes_per_dimensions;
};

} // namespace infrastructure

#endif
