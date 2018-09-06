/**
 * utilfuncs.h: an assortment of convenience functions for channelflow/programs
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef BASICS_ARGLIST_H
#define BASICS_ARGLIST_H

#include "cfbasics/cfarray.h"
#include "cfbasics/cfbasics.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

namespace chflow {

// A simple way to get flexible command-line args for chflow programs
// Scales as N^2 with # args, so don't use for hundreds of args.

class ArgList {
   public:
    typedef std::string str;  // reduces line-length below

    inline ArgList();
    inline ArgList(int argc, char* argv[], const str& purpose);

    inline bool helpmode() const;
    inline bool errormode() const;
    inline int remaining() const;                                             // how many args have yet to be parsed
    inline void section(const str& name, const str& description = "") const;  // print section header in help mode

    // return true if the option is in the args list, false if not
    inline bool getflag(const str& shortopt, const str& longopt, const str& helpstr);

    inline bool getbool(const str& shortopt, const str& longopt, const str& helpstr);
    inline int getint(const str& shortopt, const str& longopt, const str& helpstr);
    inline Real getreal(const str& shortopt, const str& longopt, const str& helpstr);
    inline str getstr(const str& shortopt, const str& longopt, const str& helpstr);
    inline str getpath(const str& shortopt, const str& longopt, const str& helpstr);

    inline bool getbool(const str& shortopt, const str& longopt, bool defalt, const str& helpstr);
    inline int getint(const str& shortopt, const str& longopt, int defalt, const str& helpstr);
    inline Real getreal(const str& shortopt, const str& longopt, Real defalt, const str& helpstr);
    inline str getstr(const str& shortopt, const str& longopt, const str& defalt, const str& helpstr);
    inline str getpath(const str& shortopt, const str& longopt, const str& defalt, const str& helpstr);

    // In the following position counts backwards from end of arglist, e.g
    // in "command arg3 arg2 arg1" the args are numbered as indicated.
    inline Real getreal(int position, const str& meaning, const str& helpstr);
    inline str getpath(int position, const str& meaning, const str& helpstr);
    inline str getstr(int position, const str& meaning, const str& helpstr);

    // Return all arguments after the last one that is already used
    // Allows for giving a list of files like 'command -a 0 -b 1 *.h5'
    inline std::vector<std::string> remainingatend();

    inline void save(const str& outdir) const;  // save command-line to file <argv[0]>.args
    inline void save() const;                   // save command-line to file <argv[0]>.args
    inline void check();                        // check for unrecognized options and arguments

   private:
    cfarray<str> args_;
    cfarray<bool> used_;
    bool helpmode_;
    bool errormode_;
    inline void printhelp(const str& sopt, const str& lopt, const str& type, const str& defalt, const str& helpstr);
    inline void printhelp(int position, const str& name, const str& helpstr);
};

// If s is numeric, convert to real using atof
// If s is alpha, try to open file s.asc
inline Real arg2real(const std::string& s) {
    Real rtn;
    if (fileExists(s))
        load(rtn, s);
    else
        rtn = atof(s.c_str());
    return rtn;
}

inline ArgList::ArgList() : args_(), used_(), helpmode_(false), errormode_(false) {}

inline ArgList::ArgList(int argc, char* argv[], const std::string& purpose)
    : args_(argc), used_(argc), helpmode_(false), errormode_(false) {
    std::string h0("-h");
    std::string h1("--help");
    for (int i = 0; i < argc; ++i) {
        args_[i] = std::string(argv[i]);
        if (args_[i] == h0 || args_[i] == h1) {
            helpmode_ = true;
            used_[i] = true;
        } else
            used_[i] = false;
    }
    if (helpmode_) {
        std::cerr << argv[0] << " : \n\t" << purpose << std::endl << std::endl;
    }
    used_[0] = true;
}

inline bool ArgList::helpmode() const { return helpmode_; }

inline bool ArgList::errormode() const { return errormode_; }

inline int ArgList::remaining() const {
    int unused = 0;
    for (int i = 0; i < used_.length(); ++i)
        if (!used_[i])
            ++unused;
    return unused;
}

inline void ArgList::section(const str& name, const str& description) const {
    if (helpmode_) {
        std::cerr << std::endl << name << ":" << std::endl;
        if (description.size() > 0) {
            std::cerr << description << std::endl;
        }
    }
}

inline std::vector<std::string> ArgList::remainingatend() {
    int n = args_.length() - 1;
    while (n > 0 && used_[n] == false)
        n--;

    std::vector<std::string> result;
    while (n < args_.length() - 1) {
        n++;
        result.push_back(args_[n]);
        used_[n] = true;
    }
    return result;
}

// The magic setw constants are a quick and dirty way to line up the columns
// nicely as long as the strings aren't too long. If you want to reimplement
// formatting more intelligently, please do!
inline void ArgList::printhelp(int position, const std::string& name, const std::string& helpstr) {
    std::cerr.setf(std::ios::left);
    std::cerr << "  ";
    std::cerr << std::setw(17) << name;
    std::cerr << std::setw(48) << std::string("(trailing arg " + i2s(position) + ")");
    std::cerr.unsetf(std::ios::left);
    std::cerr << helpstr << std::endl;
}

inline void ArgList::printhelp(const std::string& sopt, const std::string& lopt, const std::string& type,
                               const std::string& defalt, const std::string& helpstr) {
    std::cerr.setf(std::ios::left);
    std::cerr << "  " << std::setw(8) << sopt << "  ";
    std::cerr << std::setw(20) << lopt << std::setw(10) << type;

    if (defalt.length() != 0)
        std::cerr << "  default == " << std::setw(12) << defalt;
    else
        std::cerr << std::setw(25) << "";
    std::cerr.unsetf(std::ios::left);
    std::cerr << helpstr << std::endl;
}

inline bool ArgList::getflag(const std::string& sopt, const std::string& lopt, const std::string& helpstr) {
    if (helpmode_) {
        printhelp(sopt, lopt, "", "", helpstr);
        return false;
    }

    bool b = false;
    int N = args_.length();
    for (int n = N - 1; n >= 1; --n) {
        if (args_[n] == sopt || args_[n] == lopt) {
            b = true;
            used_[n] = true;
            break;
        }
    }
    return b;
}

inline bool ArgList::getbool(const std::string& sopt, const std::string& lopt, const std::string& helpstr) {
    bool b = false;

    int N = args_.length();
    bool found = false;
    for (int n = N - 1; n >= 1; --n) {
        if (args_[n] == sopt || args_[n] == lopt) {
            found = true;
            if (n < N - 1 && args_[n + 1] == "true")
                b = true;
            else if (n < N - 1 && args_[n + 1] == "false")
                b = false;
            else {
                std::cerr << "error : option " << sopt << " or " << lopt << " should be followed by 'true' or 'false'."
                          << std::endl;
                exit(1);
            }
            used_[n] = true;
            used_[n + 1] = true;
            break;
        }
    }
    if (helpmode_) {
        printhelp(sopt, lopt, "<bool>", "", helpstr);
        return b;
    } else if (!found) {
        errormode_ = true;
        std::cerr << "Missing required argument: " << sopt << " or " << lopt << std::endl;
    }
    return b;
}

inline int ArgList::getint(const std::string& sopt, const std::string& lopt, const std::string& helpstr) {
    int i = 0;
    bool found = false;
    int N = args_.length();
    for (int n = N - 1; n >= 1; --n) {
        if (args_[n] == sopt || args_[n] == lopt) {
            found = true;
            if (n < N - 1)
                i = atoi(args_[n + 1].c_str());
            else {
                std::cerr << "error : option " << sopt << " or " << lopt << " should be followed by an integer."
                          << std::endl;
                exit(1);
            }
            used_[n] = true;
            used_[n + 1] = true;
            break;
        }
    }
    if (helpmode_) {
        printhelp(sopt, lopt, "<int>", "", helpstr);
        return i;
    } else if (!found) {
        errormode_ = true;
        std::cerr << "Missing required argument: " << sopt << " or " << lopt << std::endl;
    }
    return i;
}

inline Real ArgList::getreal(const std::string& sopt, const std::string& lopt, const std::string& helpstr) {
    Real r = 0.0;

    int N = args_.length();
    bool found = false;
    for (int n = N - 1; n >= 1; --n) {
        if (args_[n] == sopt || args_[n] == lopt) {
            found = true;
            if (n < N - 1)
                // r = atof(args_[n+1].c_str());
                r = arg2real(args_[n + 1].c_str());
            else {
                std::cerr << "error : option " << sopt << " or " << lopt << " should be followed by a real number."
                          << std::endl;
                exit(1);
            }
            used_[n] = true;
            used_[n + 1] = true;
            break;
        }
    }
    if (helpmode_) {
        printhelp(sopt, lopt, "<real>", "", helpstr);
        return r;
    } else if (!found) {
        errormode_ = true;
        std::cerr << "Missing required argument: " << sopt << " or " << lopt << std::endl;
    }
    return r;
}

inline std::string ArgList::getpath(const std::string& sopt, const std::string& lopt, const std::string& helpstr) {
    return pathfix(getstr(sopt, lopt, helpstr));
}

inline std::string ArgList::getstr(const std::string& sopt, const std::string& lopt, const std::string& helpstr) {
    std::string s("");

    bool found = false;
    int N = args_.length();
    for (int n = N - 1; n >= 1; --n) {
        if (args_[n] == sopt || args_[n] == lopt) {
            found = true;
            if (n < N - 1)
                s = std::string(args_[n + 1]);
            else {
                std::cerr << "error : option " << sopt << " or " << lopt << " should be followed by a string."
                          << std::endl;
                exit(1);
            }
            used_[n] = true;
            used_[n + 1] = true;
            break;
        }
    }
    if (helpmode_) {
        printhelp(sopt, lopt, "<string>", "", helpstr);
        return s;
    } else if (!found) {
        errormode_ = true;
        std::cerr << "Missing required argument: " << sopt << " or " << lopt << std::endl;
    }
    return s;
}

inline bool ArgList::getbool(const std::string& sopt, const std::string& lopt, bool defalt,
                             const std::string& helpstr) {
    bool b = defalt;
    if (helpmode_) {
        printhelp(sopt, lopt, "<bool>", std::string(b ? "true" : "false"), helpstr);
        return b;
    }

    int N = args_.length();
    for (int n = N - 1; n >= 1; --n) {
        if (args_[n] == sopt || args_[n] == lopt) {
            if (n < N - 1 && args_[n + 1] == "true")
                b = true;
            else if (n < N - 1 && args_[n + 1] == "false")
                b = false;
            else {
                std::cerr << "error : option " << sopt << " or " << lopt << " should be followed by 'true' or 'false'."
                          << std::endl;
                exit(1);
            }
            used_[n] = true;
            used_[n + 1] = true;
            break;
        }
    }
    return b;
}

inline int ArgList::getint(const std::string& sopt, const std::string& lopt, int defalt, const std::string& helpstr) {
    int i = defalt;
    if (helpmode_) {
        printhelp(sopt, lopt, "<int>", i2s(defalt), helpstr);
        return i;
    }
    int N = args_.length();
    for (int n = N - 1; n >= 1; --n) {
        if (args_[n] == sopt || args_[n] == lopt) {
            if (n < N - 1)
                i = atoi(args_[n + 1].c_str());
            else {
                std::cerr << "error : option " << sopt << " or " << lopt << " should be followed by an integer."
                          << std::endl;
                exit(1);
            }
            used_[n] = true;
            used_[n + 1] = true;
            break;
        }
    }
    return i;
}

inline Real ArgList::getreal(const std::string& sopt, const std::string& lopt, Real defalt,
                             const std::string& helpstr) {
    Real r = defalt;
    if (helpmode_) {
        printhelp(sopt, lopt, "<real>", r2s(defalt), helpstr);
        return r;
    }

    int N = args_.length();
    for (int n = N - 1; n >= 1; --n) {
        if (args_[n] == sopt || args_[n] == lopt) {
            if (n < N - 1)
                // r = atof(args_[n+1].c_str());
                r = arg2real(args_[n + 1].c_str());
            else {
                std::cerr << "error : option " << sopt << " or " << lopt << " should be followed by a real number."
                          << std::endl;
                exit(1);
            }
            used_[n] = true;
            used_[n + 1] = true;
            break;
        }
    }
    return r;
}

inline std::string ArgList::getpath(const std::string& sopt, const std::string& lopt, const std::string& defalt,
                                    const std::string& helpstr) {
    return pathfix(getstr(sopt, lopt, defalt, helpstr));
}

inline std::string ArgList::getstr(const std::string& sopt, const std::string& lopt, const std::string& defalt,
                                   const std::string& helpstr) {
    std::string s(defalt);
    if (helpmode_) {
        printhelp(sopt, lopt, "<string>", defalt, helpstr);
        return defalt;
    }
    int N = args_.length();
    for (int n = N - 1; n >= 1; --n) {
        if (args_[n] == sopt || args_[n] == lopt) {
            if (n < N - 1)
                s = std::string(args_[n + 1]);
            else {
                std::cerr << "error : option " << sopt << " or " << lopt << " should be followed by a string."
                          << std::endl;
                exit(1);
            }
            used_[n] = true;
            used_[n + 1] = true;
            break;
        }
    }
    return s;
}

inline std::string ArgList::getstr(int position, const std::string& name, const std::string& helpstr) {
    if (helpmode_) {
        printhelp(position, name, helpstr);
        return std::string();
    }

    int n = args_.length() - position;
    if (n < 1 || used_[n]) {
        std::cerr << "error : " << name << " is required as the " << position
                  << "th trailing argument (counting backwards from end of arglist)" << std::endl;
        ;
        exit(1);
    }
    if (args_[n][0] == '-') {
        std::cerr << "error : should have a filename for " << position << "th trailing argument, ";
        std::cerr << "not option " << args_[n] << std::endl;
        exit(1);
    }
    used_[n] = true;
    return args_[n];
}

inline std::string ArgList::getpath(int position, const std::string& name, const std::string& helpstr) {
    return pathfix(getstr(position, name, helpstr));
}

inline Real ArgList::getreal(int position, const std::string& name, const std::string& helpstr) {
    if (helpmode_) {
        printhelp(position, name, helpstr);
        return 0.0;
    }

    int n = args_.length() - position;
    if (n < 1 || used_[n]) {
        std::cerr << "error : " << name << " is required as the " << position
                  << "th trailing argument (counting backwards from end of arglist)" << std::endl;
        ;
        exit(1);
    }
    used_[n] = true;
    return arg2real(args_[n].c_str());
}

inline void ArgList::check() {
    for (int n = 0; n < used_.length(); ++n) {
        if (!used_[n]) {
            std::cerr << "error : unrecognized/repeated option/value " << args_[n] << std::endl;
            errormode_ = true;
        }
    }
    if (errormode_) {
        std::cerr << "Error in specifying program arguments. Please rerun program with -h option " << std::endl;
        std::cerr << "and review the argument specifications. Exiting." << std::endl;
        exit(1);
    } else if (helpmode_) {
        exit(0);
    }
}

inline void ArgList::save() const { this->save("./"); }

inline void ArgList::save(const std::string& outdir) const {
    if (mpirank() == 0) {
        // Look for output directory and save command-line to outdir/argv[0].args
        std::string arg0 = args_[0];
        std::string exec;
        int slashpos = arg0.find_last_of('/');
        if (slashpos == -1) {
            exec = arg0;
        } else {
            exec = std::string(arg0, slashpos + 1, arg0.length());
        }
        int extpos = exec.find(".x");
        if (extpos == -1)
            extpos = exec.find(".dx");
        if (extpos == -1)
            extpos = exec.length();

        std::string arse = std::string(exec, 0, extpos) + ".args";
        std::string afile = outdir + arse;
        ::mkdir(outdir.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        std::ofstream as(afile.c_str(), std::ios::app);

        // Four statements, and two types, and two possible memory leaks to get the current time.
        // Ain't C wonderful?
        time_t now;
        time(&now);                 // get the calendar time
        tm* t = localtime(&now);    // convert to local (memory leak?)
        std::string s(asctime(t));  // memory leak?
        s.erase(s.length() - 1);    // remove ridiculous newline from asctime output
        as << s << '\t';

        // get process ID. equally wonderful
        as << getpid() << '\t';

        for (int n = 0; n < args_.length(); ++n)
            as << args_[n] << ' ';
        as << std::endl;
    }
}

}  // namespace chflow
#endif
