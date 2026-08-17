//utilities.c
#include <ctype.h>
#include <string.h>
#include "utilities.h"

const char* square_to_algebraic(int sq) {
    static char buf[8][3];  // allow 8 concurrent calls
    static int idx = 0;
    idx = (idx + 1) & 7;
    int file = sq % 8;
    int rank = sq / 8;
    buf[idx][0] = 'a' + file;
    buf[idx][1] = '1' + rank;
    buf[idx][2] = '\0';
    return buf[idx];
}