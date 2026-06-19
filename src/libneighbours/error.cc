/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#include "error.hh"

#include <cstring>

namespace matscipy {

bool has_error = false;
char error_string[MAX_ERROR_STRING] = {0};
bool has_warning = false;
char warning_string[MAX_ERROR_STRING] = {0};

static void copy_message(char *dst, const char *msg) {
    std::strncpy(dst, msg, MAX_ERROR_STRING - 1);
    dst[MAX_ERROR_STRING - 1] = '\0';
}

error_t set_error(const char *msg) {
    has_error = true;
    copy_message(error_string, msg);
    return NL_ERROR;
}

void set_warning(const char *msg) {
    has_warning = true;
    copy_message(warning_string, msg);
}

void clear_error() {
    has_error = false;
    has_warning = false;
}

}  // namespace matscipy
