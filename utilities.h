#ifndef UTILITIES_H
#define UTILITIES_H

#ifdef __GNUC__
#define likely(x) __builtin_expect((x),1)
#define unlikely(x) __builtin_expect((x),0)
#else
#define likely(x)   (x)
#define unlikely(x) (x)
#endif

// Converts a square index [0..63] into algebraic notation ("a1".."h8")
const char* square_to_algebraic(int sq);

#endif
