#include <cstdint>
#include <cassert>
#include <algorithm>
#include "tipsy.h"

TipsyIO::TipsyIO() {
    buffer = std::make_unique<char[]>(buffer_size);
    }

TipsyIO::TipsyIO(const        char* filename,  std::ios_base::openmode mode) {
    buffer = std::make_unique<char[]>(buffer_size);
    open(filename,mode);
    }

TipsyIO::TipsyIO(const std::string& filename,  std::ios_base::openmode mode) {
    buffer = std::make_unique<char[]>(buffer_size);
    open(filename,mode);
    }

void TipsyIO::open (const char* filename,  std::ios_base::openmode mode) {
    std::ifstream::open(filename,mode|std::ios_base::binary);
    if (fail()) return;

    // Read in the Tipsy header
    tipsyio::tipsyHdr hdr;
    std::ifstream::read(reinterpret_cast<char*>(&hdr),sizeof(hdr));
    if (fail()) { close(); return; }

    // First, zero is not a valid number of dimensions
    if (hdr.nDim==0) { setstate(std::ios::failbit); close(); return; }

    // Now check to see if this is a "standard" format file
    if (hdr.nDim>3) {
        tipsyio::swap(hdr);
        // If we are still inconsistent then give up
        if (hdr.nDim>3) { setstate(std::ios::failbit); close(); return; }
        bStandard = true;
        }
    else bStandard = false;
    nSph  = hdr.nSph;
    nDark = hdr.nDark;
    nStar = hdr.nStar;
    }

void TipsyIO::open (const std::string& filename,  std::ios_base::openmode mode) {
    open(filename.c_str(),mode);
    }

// Given a particle index, seek to the correct place in the file
void TipsyIO::seekpos(std::uint64_t i) {
    uint64_t offset = sizeof(tipsyio::tipsyHdr), n;
    n = std::min(i,nSph);  offset += n * sizeof(tipsyio::tipsySph);  i -= n;
    n = std::min(i,nDark); offset += n * sizeof(tipsyio::tipsyDark); i -= n;
    n = std::min(i,nStar); offset += n * sizeof(tipsyio::tipsyStar); i -= n;
    assert(i==0);
    seekg(offset);
    }