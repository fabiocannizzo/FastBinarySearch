#pragma once

template <class T>
struct AlignedVec
{
    static const size_t A = 64;  // size of cache line

    AlignedVec()
        : m_data(0)
        , m_sz(0)
    {
    }

    void resize(size_t sz)
    {
        unsigned char *p = new unsigned char[sz * sizeof(T) + A];
        unsigned char offset = static_cast<unsigned char>(A - reinterpret_cast<size_t>(p) % A);

        p += offset;
        p[-1] = offset;

        m_data = reinterpret_cast<T*>(p);
        m_sz = sz;
    }

    ~AlignedVec()
    {
        unsigned char *p = reinterpret_cast<unsigned char *>(m_data);
        delete[](p - p[-1]);
        m_data = 0;
    }

    size_t size() const { return m_sz; }
    T& operator[](size_t i) { return m_data[i]; }
    const T& operator[](size_t i) const { return m_data[i]; }
    T* begin()  { return &m_data[0];  }
    T* end()  { return &m_data[0]+m_sz; }
    T& front() { return m_data[0]; }
    T& back() { return m_data[m_sz-1]; }
    const T& front() const { return m_data[0]; }
    const T& back() const { return m_data[m_sz - 1]; }

private:
    T *m_data;
    size_t m_sz;
};
