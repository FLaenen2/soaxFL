#ifndef iterator_hpp
#define iterator_hpp

#include <memory>
#include "./macros.hpp"
#ifdef __BOOST_COMPUTE__
	#include <boost/compute/buffer.hpp>
#endif

namespace SOAX {

template <class Expr_>
class CIterator {
public:

  typedef typename std::remove_reference<Expr_>::type Expr;

  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using reference = typename Expr::value_type&;
  using pointer = typename Expr::value_type*;
  using value_type = typename Expr::value_type;

  CIterator() = default;
  CIterator(const Expr& e, difference_type i) :
    expr_(&e), i_(i) 
  {
    //    std::cerr << "  CIterator(const Expr& e, difference_type i)\n";
  }

  CIterator(const Expr&& e, difference_type i) :
    exprTemp_(new Expr(e)), expr_(&e), i_(i) { 
    //    std::cerr << "  CIterator(const Expr&& e, difference_type i)\n";
  }
  
  bool operator==(const CIterator& that) const {
    return i_ == that.i_;
  }
  
  bool operator!=(const CIterator& that) const {
    return !(*this == that);
  }
  // Similarly for operators <, >, <=, >=

  value_type operator*() const {
    return (*expr_)[i_];
  }

  value_type operator[](difference_type n) const {
    return (*expr_)[i_ + n];
  }

  CIterator& operator=(const CIterator& it)
  {
    expr_ = it.expr_;
    i_ = it.i_;
    return *this;
  }

  CIterator& operator++() { 
    ++i_; return *this; 
  }

  CIterator operator++(int) { 
    auto tmp = *this; ++i_; 
    return tmp; 
  }

  CIterator& operator--() { --i_; return *this; }
  CIterator operator--(int) { auto tmp = *this; --*this; return tmp; }

  CIterator operator+(difference_type n) const {
    return CIterator{*expr_, i_ + n};
  }
  CIterator operator-(difference_type n) const {
    return CIterator{expr_, i_ - n};
  }

  difference_type operator-(const CIterator& it) const {
    return i_ - it.i_;
  }

  // Similarly for operators +=, and -=

  friend CIterator operator+(difference_type n, const CIterator& i) {
    return i + n;
  }

  friend CIterator operator-(difference_type n, const CIterator& i) {
    return i - n;
  }

private:

  const Expr* exprTemp_;
  const Expr* expr_;
  
  difference_type i_;
};

template <class Expr_>
class Iterator {
public:
  typedef typename std::remove_reference<Expr_>::type Expr;
  
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using reference = typename Expr::value_type&;
  using pointer = typename Expr::value_type*;
  using value_type = typename Expr::value_type;
  typedef float type;

  Iterator() = default;
  Iterator(Expr& e, difference_type i) :
    expr_(&e), i_(i) {
    DEBUGMSG("  Iterator(Expr& e, difference_type i)\n");
  }

  Iterator(Expr&& e, difference_type i) :
    exprTemp_(new Expr(e)), expr_(&e), i_(i) { 
    DEBUGMSG("  Iterator(Expr&& e, difference_type i)\n");
  }

  Iterator& operator=(const Iterator& it)
  {
    expr_ = it.expr_;
    i_ = it.i_;
    return *this;
  }

  bool operator==(const Iterator& that) const {
    return i_ == that.i_;
  }
  
  bool operator!=(const Iterator& that) const {
    return !(*this == that);
  }
  // Similarly for operators <, >, <=, >=
#ifdef __NVCC__
  thrust::device_reference<float> operator*() {
#elif defined __BOOST_COMPUTE__
 boost::compute::buffer_value<float> operator*() {
#else
  float& operator*() {
#endif
    DEBUGMSG("  thrust::device_reference<float> operator*()  " << i_ << "\n");
    return (*expr_)[i_];
  }

  value_type operator[](difference_type n) const {
    return (*expr_)[i_ + n];
  }

  Iterator& operator++() { ++i_; return *this; }
  Iterator operator++(int) { auto tmp = *this; ++*this; return tmp;}
  Iterator& operator--() { --i_; return *this; }
  Iterator operator--(int) { auto tmp = *this; --*this; return tmp; }

  Iterator operator+(difference_type n) const {
    return Iterator{expr_, i_ + n};
  }
  Iterator operator-(difference_type n) const {
    return Iterator{expr_, i_ - n};
  }
  // Similarly for operators +=, and -=

  friend Iterator operator+(difference_type n, const Iterator& i) {
    return i + n;
  }

  friend Iterator operator-(difference_type n, const Iterator& i) {
    return i - n;
  }

  difference_type operator-(const Iterator& it) const {
    return i_ - it.i_;
  }

private:
  Expr* exprTemp_;
  Expr* expr_;
  

  
  difference_type i_;
};

} // namespace SOAX

#endif
