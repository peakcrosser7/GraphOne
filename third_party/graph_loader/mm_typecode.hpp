#pragma once

#include<string>
#include<cstring>

#include "exception.hpp"

namespace graph_loader {

/************** Matrix Market internal definitions **************************/
/**
 *                  MM_matrix_typecode: 4-character sequence
 *
 * |                 | ojbect   | sparse/dense | data type | storage scheme |
 * |-----------------|----------|--------------|-----------|----------------|
 * | string position | [0]      | [1]          | [2]       | [3]            |
 * | Matrix typecode | M(atrix) | C(oord)      | R(eal)    | G(eneral)      |
 * |                 |          | A(rray)      | C(omplex) | H(ermitian)    |
 * |                 |          |              | P(attern) | S(ymmetric)    |
 * |                 |          |              | I(nteger) | K(kew)         |
 *
 */

class MMTypecode {
public:
    static MMTypecode FromString(const std::string& line) {
        constexpr const char* kBanner = "%%MatrixMarket";
        constexpr const int kMaxTokenLen = 64;
        constexpr const char* kMatrixStr = "matrix";
        // constexpr const char* kArrayStr = "array";
        constexpr const char* kDenseStr = "array";
        // constexpr const char* kCoordinateStr = "coordinate";
        constexpr const char* kSparseStr = "coordinate";
        constexpr const char* kComplexStr = "complex";
        constexpr const char* kRealStr = "real";
        constexpr const char* kIntStr = "integer";
        constexpr const char* kGeneralStr = "general";
        constexpr const char* kSymmStr = "symmetric";
        constexpr const char* kHermStr = "hermitian";
        constexpr const char* kSkewStr = "skew-symmetric";
        constexpr const char* kPatternStr = "pattern";

        char banner[kMaxTokenLen];
        char mtx[kMaxTokenLen];
        char crd[kMaxTokenLen];
        char data_type[kMaxTokenLen];
        char storage_scheme[kMaxTokenLen];
        char *p;

        MMTypecode meta;

        int ret = std::sscanf(line.c_str(), "%s %s %s %s %s", banner, mtx, crd, data_type,
                storage_scheme);
        throw_if_exception(ret != 5, "MatrixMeta::FromString(): premature EOF");

        for (p = mtx; *p != '\0'; *p = std::tolower(*p), ++p);
        for (p = crd; *p != '\0'; *p = std::tolower(*p), ++p);
        for (p = data_type; *p != '\0'; *p = std::tolower(*p), ++p);
        for (p = storage_scheme; *p != '\0'; *p = std::tolower(*p), ++p);

        /* check for banner */
        ret = std::strncmp(banner, kBanner, std::strlen(kBanner));
        throw_if_exception (ret != 0, "MatrixMeta::FromString(): no header");

        /* first field should be "mtx" */
        ret = std::strcmp(mtx, kMatrixStr);
        throw_if_exception(ret != 0, "MatrixMeta::FromString(): unsupported type for 1st field");
        meta.set_matrix();

        /* second field describes whether this is a sparse matrix (in coordinate
                storgae) or a dense array */
        if (std::strcmp(crd, kSparseStr) == 0) {
            meta.set_sparse();
        } else if (std::strcmp(crd, kDenseStr) == 0) {
            meta.set_dense();
        } else {
            throw_if_exception(true, "MatrixMeta::FromString(): unsupported type for 2nd field");
        }

        /* third field */
        if (std::strcmp(data_type, kRealStr) == 0) {
            meta.set_real();
        } else if (std::strcmp(data_type, kComplexStr) == 0) {
            meta.set_complex();
        } else if (std::strcmp(data_type, kPatternStr) == 0) {
            meta.set_pattern();
        } else if (std::strcmp(data_type, kIntStr) == 0) {
            meta.set_integer();
        } else {
            throw_if_exception(true, "MatrixMeta::FromString(): unsupported type for 3rd field");
        }

        /* fourth field */
        if (std::strcmp(storage_scheme, kGeneralStr) == 0) {
            meta.set_general();
        } else if (std::strcmp(storage_scheme, kSymmStr) == 0) {
            meta.set_symmetric();
        } else if (std::strcmp(storage_scheme, kHermStr) == 0) {
            meta.set_hermitian();
        } else if (std::strcmp(storage_scheme, kSkewStr) == 0) {
            meta.set_skew_symmetric();
        } else {
            throw_if_exception(true, "MatrixMeta::FromString(): unsupported type for 4th field");
        }

        return meta;
    }

    MMTypecode() { clear(); }

    bool is_matrix() const { return code_[0] == 'M'; }

    bool is_sparse() const { return code_[1] == 'C'; }
    bool is_coordinate() const { return code_[1] == 'C'; }
    bool is_dense() const { return code_[1] == 'A'; }
    bool is_array() const { return code_[1] == 'A'; }

    bool is_complex() const { return code_[2] == 'C'; }
    bool is_real() const { return code_[2] == 'R'; }
    bool is_integer() const { return code_[2] == 'I'; }
    bool is_pattern() const { return code_[2] == 'P'; }

    bool is_symmetric() const { return code_[3] == 'S'; }
    bool is_general() const { return code_[3] == 'G'; }
    bool is_skew_symmetric() const { return code_[3] == 'K'; }
    bool is_hermitian() const { return code_[3] == 'H'; }

    void set_matrix() { code_[0] = 'M'; }
    void set_coordinate() { code_[1] = 'C'; }
    void set_array() { code_[1] = 'A'; }
    void set_dense() { set_array(); }
    void set_sparse() { set_coordinate(); }

    void set_complex() { code_[2] = 'C'; }
    void set_real() { code_[2] = 'R'; }
    void set_pattern() { code_[2] = 'P'; }
    void set_integer() { code_[2] = 'I'; }

    void set_symmetric() { code_[3] = 'S'; }
    void set_general() { code_[3] = 'G'; }
    void set_skew_symmetric() { code_[3] = 'K'; }
    void set_hermitian() { code_[3] = 'H'; }

    void clear() {
        code_[0] = code_[1] = code_[2] = ' ';
        code_[3] = 'G';
    }

private:
    char code_[4];
};
    
} // namespace graph_loader
