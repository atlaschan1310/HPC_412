#include <fstream>
#include <cstdint>
#include <memory>
#include "blitz/array.h"

namespace baseio {
inline void swap(uint64_t &value) {
    value =
      ((value & 0xFF00000000000000u) >> 56u) |
      ((value & 0x00FF000000000000u) >> 40u) |
      ((value & 0x0000FF0000000000u) >> 24u) |
      ((value & 0x000000FF00000000u) >>  8u) |
      ((value & 0x00000000FF000000u) <<  8u) |
      ((value & 0x0000000000FF0000u) << 24u) |
      ((value & 0x000000000000FF00u) << 40u) |
      ((value & 0x00000000000000FFu) << 56u);
    }
inline void swap(double &d) {
    uint64_t i;
    static_assert(sizeof i == sizeof d);
    std::memcpy(&i,&d,sizeof i);
    swap(i);
    std::memcpy(&d,&i,sizeof i);
    }

inline void swap(std::uint32_t &value) {
    value =
      ((value & 0xFF000000u) >> 24u) |
      ((value & 0x00FF0000u) >>  8u) |
      ((value & 0x0000FF00u) <<  8u) |
      ((value & 0x000000FFu) << 24u);
    }

inline void swap(float &f) {
    uint32_t i;
    static_assert(sizeof i == sizeof f);
    std::memcpy(&i,&f,sizeof i);
    swap(i);
    std::memcpy(&f,&i,sizeof i);
    }
}

namespace tipsyio {
// Header of a Tipsy file
typedef struct {
    double dTime;
    std::uint32_t nBodies;
    std::uint32_t nDim;
    std::uint32_t nSph;
    std::uint32_t nDark;
    std::uint32_t nStar;
    std::uint32_t nPad;
    } tipsyHdr;
inline void swap(tipsyHdr &hdr) {
    baseio::swap(hdr.dTime);
    baseio::swap(hdr.nBodies);
    baseio::swap(hdr.nDim);
    baseio::swap(hdr.nSph);
    baseio::swap(hdr.nDark);
    baseio::swap(hdr.nStar);
    baseio::swap(hdr.nPad);
    }

// SPH (gas) particle
typedef struct {
    float mass;
    float pos[3];
    float vel[3];
    float rho;
    float temp;
    float hsmooth;
    float metals;
    float phi;
    } tipsySph;
inline void swap(tipsySph *p) {
    baseio::swap(p->mass);
    baseio::swap(p->pos[0]);
    baseio::swap(p->pos[1]);
    baseio::swap(p->pos[2]);
    baseio::swap(p->vel[0]);
    baseio::swap(p->vel[1]);
    baseio::swap(p->vel[2]);
    baseio::swap(p->rho);
    baseio::swap(p->temp);
    baseio::swap(p->hsmooth);
    baseio::swap(p->metals);
    baseio::swap(p->phi);
    }

// Dark matter particle
typedef struct {
    float mass;
    float pos[3];
    float vel[3];
    float eps;
    float phi;
    } tipsyDark;
inline void swap(tipsyDark *p) {
    baseio::swap(p->mass);
    baseio::swap(p->pos[0]);
    baseio::swap(p->pos[1]);
    baseio::swap(p->pos[2]);
    baseio::swap(p->vel[0]);
    baseio::swap(p->vel[1]);
    baseio::swap(p->vel[2]);
    baseio::swap(p->eps);
    baseio::swap(p->phi);
    }

// Star particle
typedef struct {
    float mass;
    float pos[3];
    float vel[3];
    float metals;
    float tform;
    float eps;
    float phi;
    } tipsyStar;
inline void swap(tipsyStar *p) {
    baseio::swap(p->mass);
    baseio::swap(p->pos[0]);
    baseio::swap(p->pos[1]);
    baseio::swap(p->pos[2]);
    baseio::swap(p->vel[0]);
    baseio::swap(p->vel[1]);
    baseio::swap(p->vel[2]);
    baseio::swap(p->metals);
    baseio::swap(p->tform);
    baseio::swap(p->eps);
    baseio::swap(p->phi);
    }
}

class TipsyIO : protected std::ifstream {
public: // Provide some stream status functions
    using std::ifstream::good;
    using std::ifstream::fail;
    using std::ifstream::bad;
public:
    template<typename typeR=float>
    using rarray1 = blitz::Array<typeR,1>;
    template<typename typeR=float>
    using rarray2 = blitz::Array<typeR,2>;
protected:
    static constexpr size_t buffer_size = 1024*1024;
    std::unique_ptr<char[]> buffer;
    bool bStandard = false;
    std::uint64_t nSph, nDark, nStar;
    void seekpos(std::uint64_t iParticle);

    template<typename BLOCK>
    uint64_t readChunk(BLOCK *p, uint64_t i, uint64_t end) {
        uint64_t maxParticles = buffer_size / sizeof(BLOCK);
        auto numParticles = std::min(end-i,maxParticles);
        read(buffer.get(),numParticles*sizeof(BLOCK));
        if (fail()) numParticles = 0;
        return numParticles;
        }
public:
    TipsyIO();
    TipsyIO(const        char* filename,  std::ios_base::openmode mode = std::ios_base::in);
    TipsyIO(const std::string& filename,  std::ios_base::openmode mode = std::ios_base::in);
    TipsyIO(const TipsyIO&) = delete;
    virtual ~TipsyIO() = default;
    void open (const        char* filename,  std::ios_base::openmode mode = std::ios_base::in);
    void open (const std::string& filename,  std::ios_base::openmode mode = std::ios_base::in);
    uint64_t count() const { return nSph + nDark + nStar; }

    template<typename typeR=float,typename typeV=typeR,typename typeM=typeV>
    void load(rarray1<typeR> &x,rarray1<typeR> &y,rarray1<typeR> &z) {
        using blitz::firstRank;
        // Start reading at the start of the array (often/normally 0)
        uint64_t i = x.lbound(firstRank);        // Index of first particle to read
        uint64_t last = i + x.extent(firstRank); // Index of list particle + 1
        assert(y.lbound(firstRank)==i && z.lbound(firstRank)==i);
        seekpos(i);

        // Read SPH particles (if any)
        auto end = std::min(nSph,last);
        while(i<end) {
            auto p = reinterpret_cast<tipsyio::tipsySph*>(buffer.get());
            auto n = readChunk(p,i,end);
            if (fail()) return;
            assert(n>0);
            while (n--) {
                if (bStandard) tipsyio::swap(p);
                x(i) = p->pos[0];
                y(i) = p->pos[1];
                z(i) = p->pos[2];
                ++p;
                ++i;
                }
            }

        // Read Dark particles (if any)
        end = std::min(nSph+nDark,last);
        while(i<end) {
            auto p = reinterpret_cast<tipsyio::tipsyDark*>(buffer.get());
            auto n = readChunk(p,i,end);
            if (fail()) return;
            assert(n>0);
            while (n--) {
                if (bStandard) tipsyio::swap(p);
                x(i) = p->pos[0];
                y(i) = p->pos[1];
                z(i) = p->pos[2];
                ++p;
                ++i;
                }
            }

        // Read Star particles (if any)
        end = std::min(nSph+nDark+nStar,last);
        while(i<end) {
            auto p = reinterpret_cast<tipsyio::tipsyStar*>(buffer.get());
            auto n = readChunk(p,i,end);
            if (fail()) return;
            assert(n>0);
            while (n--) {
                if (bStandard) tipsyio::swap(p);
                x(i) = p->pos[0];
                y(i) = p->pos[1];
                z(i) = p->pos[2];
                ++p;
                ++i;
                }
            }
        }

    // If the position/velocity array is given as a 2D then call the 1D version
    template<typename typeR=float,typename typeV=typeR,typename typeM=typeV>
    void load(rarray2<typeR> &r) {
        using blitz::secondRank, blitz::Range;
        assert(r.base(secondRank)==0 && r.extent(secondRank)==3);
        rarray1<typeR> x = r(Range::all(),0);
        rarray1<typeR> y = r(Range::all(),1);
        rarray1<typeR> z = r(Range::all(),2);
        load(x,y,z);
        }
    };
