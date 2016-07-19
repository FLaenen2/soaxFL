#ifdef DEBUG
	#define DEBUGMSG(x) std::cerr << x
	#define DEBUGCMD(x) x
#else
	#define DEBUGMSG(x)
	#define DEBUGCMD(x)
#endif
