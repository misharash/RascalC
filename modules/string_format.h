// safer replacement for snprintf with some reasonable upper limit on the string length (often was 1000)
// from https://stackoverflow.com/a/26221725/23322509
#include <memory>
#include <string>
#include <stdexcept>

template<typename ... Args>
std::string string_format( const char* format, Args ... args ) // version with char* format (as often used) to avoid extra casting
{
    int size_s = std::snprintf( nullptr, 0, format, args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format, args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

template<typename ... Args>
inline std::string string_format( const std::string& format, Args ... args ) // original version with string format, rewritten to reuse the code above
{
    return string_format( format.c_str(), args ... );
}