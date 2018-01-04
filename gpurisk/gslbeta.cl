
/* The function beta_inc_AXPY(A,Y,a,b,x) computes A * beta_inc(a,b,x)
+ Y taking account of possible cancellations when using the
hypergeometric transformation beta_inc(a,b,x)=1-beta_inc(b,a,1-x).

It also adjusts the accuracy of beta_inc() to fit the overall
absolute error when A*beta_inc is added to Y. (e.g. if Y >>
A*beta_inc then the accuracy of beta_inc can be reduced) */

enum {
	GSL_SUCCESS = 0,
	GSL_FAILURE = -1,
	GSL_CONTINUE = -2,  /* iteration has not converged */
	GSL_EDOM = 1,   /* input domain error, e.g sqrt(-1) */
	GSL_ERANGE = 2,   /* output range error, e.g. exp(1e100) */
	GSL_EFAULT = 3,   /* invalid pointer */
	GSL_EINVAL = 4,   /* invalid argument supplied by user */
	GSL_EFAILED = 5,   /* generic failure */
	GSL_EFACTOR = 6,   /* factorization failed */
	GSL_ESANITY = 7,   /* sanity check failed - shouldn't happen */
	GSL_ENOMEM = 8,   /* malloc failed */
	GSL_EBADFUNC = 9,   /* problem with user-supplied function */
	GSL_ERUNAWAY = 10,  /* iterative process is out of control */
	GSL_EMAXITER = 11,  /* exceeded max number of iterations */
	GSL_EZERODIV = 12,  /* tried to divide by zero */
	GSL_EBADTOL = 13,  /* user specified an invalid tolerance */
	GSL_ETOL = 14,  /* failed to reach the specified tolerance */
	GSL_EUNDRFLW = 15,  /* underflow */
	GSL_EOVRFLW = 16,  /* overflow  */
	GSL_ELOSS = 17,  /* loss of accuracy */
	GSL_EROUND = 18,  /* failed because of roundoff error */
	GSL_EBADLEN = 19,  /* matrix, vector lengths are not conformant */
	GSL_ENOTSQR = 20,  /* matrix not square */
	GSL_ESING = 21,  /* apparent singularity detected */
	GSL_EDIVERGE = 22,  /* integral or series is divergent */
	GSL_EUNSUP = 23,  /* requested feature is not supported by the hardware */
	GSL_EUNIMPL = 24,  /* requested feature not (yet) implemented */
	GSL_ECACHE = 25,  /* cache limit exceeded */
	GSL_ETABLE = 26,  /* table limit exceeded */
	GSL_ENOPROG = 27,  /* iteration is not making progress towards solution */
	GSL_ENOPROGJ = 28,  /* jacobian evaluations are not improving the solution */
	GSL_ETOLF = 29,  /* cannot reach the specified tolerance in F */
	GSL_ETOLX = 30,  /* cannot reach the specified tolerance in X */
	GSL_ETOLG = 31,  /* cannot reach the specified tolerance in gradient */
	GSL_EOF = 32   /* end of file */
};

/* The maximum x such that gamma(x) is not
* considered an overflow.
*/
#define GSL_SF_GAMMA_XMAX  171.0

/* The maximum n such that gsl_sf_fact(n) does not give an overflow. */
#define GSL_SF_FACT_NMAX 170

/* The maximum n such that gsl_sf_doublefact(n) does not give an overflow. */
#define GSL_SF_DOUBLEFACT_NMAX 297

#define GSL_SQRT_DBL_MIN   1.4916681462400413e-154
#define GSL_DBL_EPSILON        2.2204460492503131e-16
#define GSL_DBL_MIN        2.2250738585072014e-308
#define GSL_SQRT_DBL_EPSILON   1.4901161193847656e-08
#define GSL_LOG_DBL_MIN   (-7.0839641853226408e+02)
#define GSL_LOG_DBL_MAX    7.0978271289338397e+02
#define GSL_SQRT_DBL_MAX   1.3407807929942596e+154
#define GSL_SQRT_DBL_EPSILON   1.4901161193847656e-08
#define GSL_ROOT3_DBL_EPSILON  6.0554544523933429e-06
#define GSL_ROOT4_DBL_EPSILON  1.2207031250000000e-04
#define GSL_ROOT5_DBL_EPSILON  7.4009597974140505e-04
#define GSL_ROOT6_DBL_EPSILON  2.4607833005759251e-03
#define GSL_LOG_DBL_EPSILON   (-3.6043653389117154e+01)

#ifndef M_PI
#define M_PI       3.14159265358979323846264338328      /* pi */
#endif

# define GSL_NEGINF (gsl_neginf())
# define GSL_POSINF (gsl_posinf())
# define GSL_NAN (gsl_nan())

/* other needlessly compulsive abstractions */

#define GSL_IS_ODD(n)  ((n) & 1)
#define GSL_IS_EVEN(n) (!(GSL_IS_ODD(n)))
#define GSL_SIGN(x)    ((x) >= 0.0 ? 1 : -1)
#ifndef M_E
#define M_E        2.71828182845904523536028747135      /* e */
#endif

#define LogRootTwoPi_  0.9189385332046727418

#ifndef M_LNPI
#define M_LNPI     1.14472988584940017414342735135      /* ln(pi) */
#endif

#ifndef M_EULER
#define M_EULER    0.57721566490153286060651209008      /* Euler constant */
#endif

#ifndef M_LN2
#define M_LN2      0.69314718055994530941723212146      /* ln(2) */
#endif

#ifndef M_SQRT2
#define M_SQRT2    1.41421356237309504880168872421      /* sqrt(2) */
#endif

#ifndef M_SQRTPI
#define M_SQRTPI   1.77245385090551602729816748334      /* sqrt(pi) */
#endif


double gsl_fdiv(const double x, const double y)
{
	return x / y;
}

double gsl_nan(void)
{
	return gsl_fdiv(0.0, 0.0);
}

double gsl_posinf(void)
{
	return gsl_fdiv(+1.0, 0.0);
}

double gsl_neginf(void)
{
	return gsl_fdiv(-1.0, 0.0);
}

double gsl_pow_3(const double x) { return x*x*x; }

void gsl_error(__constant char * file, int line, int gsl_errno)
{
	;
}

struct gsl_sf_result_struct {
	double val;
	double err;
};

typedef struct gsl_sf_result_struct gsl_sf_result;

#define GSL_ERROR(gsl_errno) \
       do { \
       gsl_error (__FILE__, __LINE__, gsl_errno) ; \
       return gsl_errno ; \
       } while (0)

#define GSL_ERROR_SELECT_2(a,b)       ((a) != GSL_SUCCESS ? (a) : ((b) != GSL_SUCCESS ? (b) : GSL_SUCCESS))
#define GSL_ERROR_SELECT_3(a,b,c)     ((a) != GSL_SUCCESS ? (a) : GSL_ERROR_SELECT_2(b,c))
#define GSL_ERROR_SELECT_4(a,b,c,d)   ((a) != GSL_SUCCESS ? (a) : GSL_ERROR_SELECT_3(b,c,d))
#define GSL_ERROR_SELECT_5(a,b,c,d,e) ((a) != GSL_SUCCESS ? (a) : GSL_ERROR_SELECT_4(b,c,d,e))

#define DOMAIN_ERROR(result) do { (result)->val = GSL_NAN; (result)->err = GSL_NAN; GSL_ERROR (GSL_EDOM); } while(0)

#define OVERFLOW_ERROR(result) do { (result)->val = GSL_POSINF; (result)->err = GSL_POSINF; GSL_ERROR (GSL_EOVRFLW); } while(0)

#define UNDERFLOW_ERROR(result) do { (result)->val = 0.0; (result)->err = GSL_DBL_MIN; GSL_ERROR (GSL_EUNDRFLW); } while(0)

/* GSL_ERROR_VAL: call the error handler, and return the given value */

#define GSL_ERROR_VAL(gsl_errno, value) \
       do { \
       gsl_error (__FILE__, __LINE__, gsl_errno) ; \
       return value ; \
       } while (0)

/* GSL_ERROR_VOID: call the error handler, and then return
(for void functions which still need to generate an error) */

#define GSL_ERROR_VOID(gsl_errno) \
       do { \
       gsl_error ( __FILE__, __LINE__, gsl_errno) ; \
       return ; \
       } while (0)

#define EVAL_RESULT(fn) \
   gsl_sf_result result; \
   int status = fn; \
   if (status != GSL_SUCCESS) { \
     GSL_ERROR_VAL(status, result.val); \
   } ; \
   return result.val;

#define EVAL_DOUBLE(fn) \
   int status = fn; \
   if (status != GSL_SUCCESS) { \
     GSL_ERROR_VAL(status, result); \
   } ; \
   return result;



#define GSL_MAX(a,b) ((a) > (b) ? (a) : (b))
#define GSL_MIN(a,b) ((a) < (b) ? (a) : (b))
#define GSL_MAX_INT(a,b)   GSL_MAX(a,b)
#define GSL_MIN_INT(a,b)   GSL_MIN(a,b)
#define GSL_MAX_DBL(a,b)   GSL_MAX(a,b)
#define GSL_MIN_DBL(a,b)   GSL_MIN(a,b)
#define GSL_MAX_LDBL(a,b)  GSL_MAX(a,b)
#define GSL_MIN_LDBL(a,b)  GSL_MIN(a,b)


#define PSI_TABLE_NMAX 100
__constant double psi_table[PSI_TABLE_NMAX + 1] = {
	0.0,/* Infinity */ /* psi(0) */
	-M_EULER,/* psi(1) */
	0.42278433509846713939348790992,/* */
	0.92278433509846713939348790992,
	1.25611766843180047272682124325,
	1.50611766843180047272682124325,
	1.70611766843180047272682124325,
	1.87278433509846713939348790992,
	2.01564147795560999653634505277,
	2.14064147795560999653634505277,
	2.25175258906672110764745616389,
	2.35175258906672110764745616389,
	2.44266167997581201673836525479,
	2.52599501330914535007169858813,
	2.60291809023222227314862166505,
	2.67434666166079370172005023648,
	2.74101332832746036838671690315,
	2.80351332832746036838671690315,
	2.86233685773922507426906984432,
	2.91789241329478062982462539988,
	2.97052399224214905087725697883,
	3.02052399224214905087725697883,
	3.06814303986119666992487602645,
	3.11359758531574212447033057190,
	3.15707584618530734186163491973,
	3.1987425128519740085283015864,
	3.2387425128519740085283015864,
	3.2772040513135124700667631249,
	3.3142410883505495071038001619,
	3.3499553740648352213895144476,
	3.3844381326855248765619282407,
	3.4177714660188582098952615740,
	3.4500295305349872421533260902,
	3.4812795305349872421533260902,
	3.5115825608380175451836291205,
	3.5409943255438998981248055911,
	3.5695657541153284695533770196,
	3.5973435318931062473311547974,
	3.6243705589201332743581818244,
	3.6506863483938174848844976139,
	3.6763273740348431259101386396,
	3.7013273740348431259101386396,
	3.7257176179372821503003825420,
	3.7495271417468059598241920658,
	3.7727829557002943319172153216,
	3.7955102284275670591899425943,
	3.8177324506497892814121648166,
	3.8394715810845718901078169905,
	3.8607481768292527411716467777,
	3.8815815101625860745049801110,
	3.9019896734278921969539597029,
	3.9219896734278921969539597029,
	3.9415975165651470989147440166,
	3.9608282857959163296839747858,
	3.9796962103242182164764276160,
	3.9982147288427367349949461345,
	4.0163965470245549168131279527,
	4.0342536898816977739559850956,
	4.0517975495308205809735289552,
	4.0690389288411654085597358518,
	4.0859880813835382899156680552,
	4.1026547480502049565823347218,
	4.1190481906731557762544658694,
	4.1351772229312202923834981274,
	4.1510502388042361653993711433,
	4.1666752388042361653993711433,
	4.1820598541888515500147557587,
	4.1972113693403667015299072739,
	4.2121367424746950597388624977,
	4.2268426248276362362094507330,
	4.2413353784508246420065521823,
	4.2556210927365389277208378966,
	4.2697055997787924488475984600,
	4.2835944886676813377364873489,
	4.2972931188046676391063503626,
	4.3108066323181811526198638761,
	4.3241399656515144859531972094,
	4.3372978603883565912163551041,
	4.3502848733753695782293421171,
	4.3631053861958823987421626300,
	4.3757636140439836645649474401,
	4.3882636140439836645649474401,
	4.4006092930563293435772931191,
	4.4128044150075488557724150703,
	4.4248526077786331931218126607,
	4.4367573696833950978837174226,
	4.4485220755657480390601880108,
	4.4601499825424922251066996387,
	4.4716442354160554434975042364,
	4.4830078717796918071338678728,
	4.4942438268358715824147667492,
	4.5053549379469826935258778603,
	4.5163439489359936825368668713,
	4.5272135141533849868846929582,
	4.5379662023254279976373811303,
	4.5486045001977684231692960239,
	4.5591308159872421073798223397,
	4.5695474826539087740464890064,
	4.5798567610044242379640147796,
	4.5900608426370772991885045755,
	4.6001618527380874001986055856
};

int gsl_sf_exp_mult_err_e(double x, double dx, double y, double dy, gsl_sf_result * result)
{
	double ay = fabs(y);

	if (y == 0.0) {
		result->val = 0.0;
		result->err = fabs(dy * exp(x));
		return GSL_SUCCESS;
	}
	else if ((x < 0.5*GSL_LOG_DBL_MAX   &&   x > 0.5*GSL_LOG_DBL_MIN)
		&& (ay < 0.8*GSL_SQRT_DBL_MAX  &&  ay > 1.2*GSL_SQRT_DBL_MIN)
		) {
		double ex = exp(x);
		result->val = y * ex;
		result->err = ex * (fabs(dy) + fabs(y*dx));
		result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
		return GSL_SUCCESS;
	}
	else {
		double ly = log(ay);
		double lnr = x + ly;

		if (lnr > GSL_LOG_DBL_MAX - 0.01) {
			OVERFLOW_ERROR(result);
		}
		else if (lnr < GSL_LOG_DBL_MIN + 0.01) {
			UNDERFLOW_ERROR(result);
		}
		else {
			double sy = GSL_SIGN(y);
			double M = floor(x);
			double N = floor(ly);
			double a = x - M;
			double b = ly - N;
			double eMN = exp(M + N);
			const double eab = exp(a + b);
			result->val = sy * eMN * eab;
			result->err = eMN * eab * 2.0*GSL_DBL_EPSILON;
			result->err += eMN * eab * fabs(dy / y);
			result->err += eMN * eab * fabs(dx);
			return GSL_SUCCESS;
		}
	}
}

/*-*-*-*-*-*-*-*-*-*-*-* Functions with Error Codes *-*-*-*-*-*-*-*-*-*-*-*/

int gsl_sf_psi_int_e(const int n, gsl_sf_result * result)
{
	/* CHECK_POINTER(result) */

	if (n <= 0) {
		DOMAIN_ERROR(result);
	}
	else if (n <= PSI_TABLE_NMAX) {
		result->val = psi_table[n];
		result->err = GSL_DBL_EPSILON * fabs(result->val);
		return GSL_SUCCESS;
	}
	else {
		/* Abramowitz+Stegun 6.3.18 */
		double c2 = -1.0 / 12.0;
		double c3 = 1.0 / 120.0;
		double c4 = -1.0 / 252.0;
		double c5 = 1.0 / 240.0;
		double ni2 = (1.0 / n)*(1.0 / n);
		double ser = ni2 * (c2 + ni2 * (c3 + ni2 * (c4 + ni2*c5)));
		result->val = log((double)n) - 0.5 / n + ser;
		result->err = GSL_DBL_EPSILON * (fabs(log((double)n)) + fabs(0.5 / n) + fabs(ser));
		result->err += GSL_DBL_EPSILON * fabs(result->val);
		return GSL_SUCCESS;
	}
}

__constant struct { int n; double f; long i; } fact_table[GSL_SF_FACT_NMAX + 1] = {
	{ 0,  1.0,     1L },
	{ 1,  1.0,     1L },
	{ 2,  2.0,     2L },
	{ 3,  6.0,     6L },
	{ 4,  24.0,    24L },
	{ 5,  120.0,   120L },
	{ 6,  720.0,   720L },
	{ 7,  5040.0,  5040L },
	{ 8,  40320.0, 40320L },

	{ 9,  362880.0,     362880L },
	{ 10, 3628800.0,    3628800L },
	{ 11, 39916800.0,   39916800L },
	{ 12, 479001600.0,  479001600L },

	{ 13, 6227020800.0,                               0 },
	{ 14, 87178291200.0,                              0 },
	{ 15, 1307674368000.0,                            0 },
	{ 16, 20922789888000.0,                           0 },
	{ 17, 355687428096000.0,                          0 },
	{ 18, 6402373705728000.0,                         0 },
	{ 19, 121645100408832000.0,                       0 },
	{ 20, 2432902008176640000.0,                      0 },
	{ 21, 51090942171709440000.0,                     0 },
	{ 22, 1124000727777607680000.0,                   0 },
	{ 23, 25852016738884976640000.0,                  0 },
	{ 24, 620448401733239439360000.0,                 0 },
	{ 25, 15511210043330985984000000.0,               0 },
	{ 26, 403291461126605635584000000.0,              0 },
	{ 27, 10888869450418352160768000000.0,            0 },
	{ 28, 304888344611713860501504000000.0,           0 },
	{ 29, 8841761993739701954543616000000.0,          0 },
	{ 30, 265252859812191058636308480000000.0,        0 },
	{ 31, 8222838654177922817725562880000000.0,       0 },
	{ 32, 263130836933693530167218012160000000.0,     0 },
	{ 33, 8683317618811886495518194401280000000.0,    0 },
	{ 34, 2.95232799039604140847618609644e38,  0 },
	{ 35, 1.03331479663861449296666513375e40,  0 },
	{ 36, 3.71993326789901217467999448151e41,  0 },
	{ 37, 1.37637530912263450463159795816e43,  0 },
	{ 38, 5.23022617466601111760007224100e44,  0 },
	{ 39, 2.03978820811974433586402817399e46,  0 },
	{ 40, 8.15915283247897734345611269600e47,  0 },
	{ 41, 3.34525266131638071081700620534e49,  0 },
	{ 42, 1.40500611775287989854314260624e51,  0 },
	{ 43, 6.04152630633738356373551320685e52,  0 },
	{ 44, 2.65827157478844876804362581101e54,  0 },
	{ 45, 1.19622220865480194561963161496e56,  0 },
	{ 46, 5.50262215981208894985030542880e57,  0 },
	{ 47, 2.58623241511168180642964355154e59,  0 },
	{ 48, 1.24139155925360726708622890474e61,  0 },
	{ 49, 6.08281864034267560872252163321e62,  0 },
	{ 50, 3.04140932017133780436126081661e64,  0 },
	{ 51, 1.55111875328738228022424301647e66,  0 },
	{ 52, 8.06581751709438785716606368564e67,  0 },
	{ 53, 4.27488328406002556429801375339e69,  0 },
	{ 54, 2.30843697339241380472092742683e71,  0 },
	{ 55, 1.26964033536582759259651008476e73,  0 },
	{ 56, 7.10998587804863451854045647464e74,  0 },
	{ 57, 4.05269195048772167556806019054e76,  0 },
	{ 58, 2.35056133128287857182947491052e78,  0 },
	{ 59, 1.38683118545689835737939019720e80,  0 },
	{ 60, 8.32098711274139014427634118320e81,  0 },
	{ 61, 5.07580213877224798800856812177e83,  0 },
	{ 62, 3.14699732603879375256531223550e85,  0 },
	{ 63, 1.982608315404440064116146708360e87,  0 },
	{ 64, 1.268869321858841641034333893350e89,  0 },
	{ 65, 8.247650592082470666723170306800e90,  0 },
	{ 66, 5.443449390774430640037292402480e92,  0 },
	{ 67, 3.647111091818868528824985909660e94,  0 },
	{ 68, 2.480035542436830599600990418570e96,  0 },
	{ 69, 1.711224524281413113724683388810e98,  0 },
	{ 70, 1.197857166996989179607278372170e100,  0 },
	{ 71, 8.504785885678623175211676442400e101,  0 },
	{ 72, 6.123445837688608686152407038530e103,  0 },
	{ 73, 4.470115461512684340891257138130e105,  0 },
	{ 74, 3.307885441519386412259530282210e107,  0 },
	{ 75, 2.480914081139539809194647711660e109,  0 },
	{ 76, 1.885494701666050254987932260860e111,  0 },
	{ 77, 1.451830920282858696340707840860e113,  0 },
	{ 78, 1.132428117820629783145752115870e115,  0 },
	{ 79, 8.946182130782975286851441715400e116,  0 },
	{ 80, 7.156945704626380229481153372320e118,  0 },
	{ 81, 5.797126020747367985879734231580e120,  0 },
	{ 82, 4.753643337012841748421382069890e122,  0 },
	{ 83, 3.945523969720658651189747118010e124,  0 },
	{ 84, 3.314240134565353266999387579130e126,  0 },
	{ 85, 2.817104114380550276949479442260e128,  0 },
	{ 86, 2.422709538367273238176552320340e130,  0 },
	{ 87, 2.107757298379527717213600518700e132,  0 },
	{ 88, 1.854826422573984391147968456460e134,  0 },
	{ 89, 1.650795516090846108121691926250e136,  0 },
	{ 90, 1.485715964481761497309522733620e138,  0 },
	{ 91, 1.352001527678402962551665687590e140,  0 },
	{ 92, 1.243841405464130725547532432590e142,  0 },
	{ 93, 1.156772507081641574759205162310e144,  0 },
	{ 94, 1.087366156656743080273652852570e146,  0 },
	{ 95, 1.032997848823905926259970209940e148,  0 },
	{ 96, 9.916779348709496892095714015400e149,  0 },
	{ 97, 9.619275968248211985332842594960e151,  0 },
	{ 98, 9.426890448883247745626185743100e153,  0 },
	{ 99, 9.332621544394415268169923885600e155,  0 },
	{ 100, 9.33262154439441526816992388563e157,  0 },
	{ 101, 9.42594775983835942085162312450e159,  0 },
	{ 102, 9.61446671503512660926865558700e161,  0 },
	{ 103, 9.90290071648618040754671525458e163,  0 },
	{ 104, 1.02990167451456276238485838648e166,  0 },
	{ 105, 1.08139675824029090050410130580e168,  0 },
	{ 106, 1.146280563734708354534347384148e170,  0 },
	{ 107, 1.226520203196137939351751701040e172,  0 },
	{ 108, 1.324641819451828974499891837120e174,  0 },
	{ 109, 1.443859583202493582204882102460e176,  0 },
	{ 110, 1.588245541522742940425370312710e178,  0 },
	{ 111, 1.762952551090244663872161047110e180,  0 },
	{ 112, 1.974506857221074023536820372760e182,  0 },
	{ 113, 2.231192748659813646596607021220e184,  0 },
	{ 114, 2.543559733472187557120132004190e186,  0 },
	{ 115, 2.925093693493015690688151804820e188,  0 },
	{ 116, 3.393108684451898201198256093590e190,  0 },
	{ 117, 3.96993716080872089540195962950e192,  0 },
	{ 118, 4.68452584975429065657431236281e194,  0 },
	{ 119, 5.57458576120760588132343171174e196,  0 },
	{ 120, 6.68950291344912705758811805409e198,  0 },
	{ 121, 8.09429852527344373968162284545e200,  0 },
	{ 122, 9.87504420083360136241157987140e202,  0 },
	{ 123, 1.21463043670253296757662432419e205,  0 },
	{ 124, 1.50614174151114087979501416199e207,  0 },
	{ 125, 1.88267717688892609974376770249e209,  0 },
	{ 126, 2.37217324288004688567714730514e211,  0 },
	{ 127, 3.01266001845765954480997707753e213,  0 },
	{ 128, 3.85620482362580421735677065923e215,  0 },
	{ 129, 4.97450422247728744039023415041e217,  0 },
	{ 130, 6.46685548922047367250730439554e219,  0 },
	{ 131, 8.47158069087882051098456875820e221,  0 },
	{ 132, 1.11824865119600430744996307608e224,  0 },
	{ 133, 1.48727070609068572890845089118e226,  0 },
	{ 134, 1.99294274616151887673732419418e228,  0 },
	{ 135, 2.69047270731805048359538766215e230,  0 },
	{ 136, 3.65904288195254865768972722052e232,  0 },
	{ 137, 5.01288874827499166103492629211e234,  0 },
	{ 138, 6.91778647261948849222819828311e236,  0 },
	{ 139, 9.61572319694108900419719561353e238,  0 },
	{ 140, 1.34620124757175246058760738589e241,  0 },
	{ 141, 1.89814375907617096942852641411e243,  0 },
	{ 142, 2.69536413788816277658850750804e245,  0 },
	{ 143, 3.85437071718007277052156573649e247,  0 },
	{ 144, 5.55029383273930478955105466055e249,  0 },
	{ 145, 8.04792605747199194484902925780e251,  0 },
	{ 146, 1.17499720439091082394795827164e254,  0 },
	{ 147, 1.72724589045463891120349865931e256,  0 },
	{ 148, 2.55632391787286558858117801578e258,  0 },
	{ 149, 3.80892263763056972698595524351e260,  0 },
	{ 150, 5.71338395644585459047893286526e262,  0 },
	{ 151, 8.62720977423324043162318862650e264,  0 },
	{ 152, 1.31133588568345254560672467123e267,  0 },
	{ 153, 2.00634390509568239477828874699e269,  0 },
	{ 154, 3.08976961384735088795856467036e271,  0 },
	{ 155, 4.78914290146339387633577523906e273,  0 },
	{ 156, 7.47106292628289444708380937294e275,  0 },
	{ 157, 1.17295687942641442819215807155e278,  0 },
	{ 158, 1.85327186949373479654360975305e280,  0 },
	{ 159, 2.94670227249503832650433950735e282,  0 },
	{ 160, 4.71472363599206132240694321176e284,  0 },
	{ 161, 7.59070505394721872907517857094e286,  0 },
	{ 162, 1.22969421873944943411017892849e289,  0 },
	{ 163, 2.00440157654530257759959165344e291,  0 },
	{ 164, 3.28721858553429622726333031164e293,  0 },
	{ 165, 5.42391066613158877498449501421e295,  0 },
	{ 166, 9.00369170577843736647426172359e297,  0 },
	{ 167, 1.50361651486499904020120170784e300,  0 },
	{ 168, 2.52607574497319838753801886917e302,  0 },
	{ 169, 4.26906800900470527493925188890e304,  0 },
	{ 170, 7.25741561530799896739672821113e306,  0 },

	/*
	{ 171, 1.24101807021766782342484052410e309,  0 },
	{ 172, 2.13455108077438865629072570146e311,  0 },
	{ 173, 3.69277336973969237538295546352e313,  0 },
	{ 174, 6.42542566334706473316634250653e315,  0 },
	{ 175, 1.12444949108573632830410993864e318,  0 },
	{ 176, 1.97903110431089593781523349201e320,  0 },
	{ 177, 3.50288505463028580993296328086e322,  0 },
	{ 178, 6.23513539724190874168067463993e324,  0 },
	{ 179, 1.11608923610630166476084076055e327,  0 },
	{ 180, 2.00896062499134299656951336898e329,  0 },
	{ 181, 3.63621873123433082379081919786e331,  0 },
	{ 182, 6.61791809084648209929929094011e333,  0 },
	{ 183, 1.21107901062490622417177024204e336,  0 },
	{ 184, 2.22838537954982745247605724535e338,  0 },
	{ 185, 4.12251295216718078708070590390e340,  0 },
	{ 186, 7.66787409103095626397011298130e342,  0 },
	{ 187, 1.43389245502278882136241112750e345,  0 },
	{ 188, 2.69571781544284298416133291969e347,  0 },
	{ 189, 5.09490667118697324006491921822e349,  0 },
	{ 190, 9.68032267525524915612334651460e351,  0 },
	{ 191, 1.84894163097375258881955918429e354,  0 },
	{ 192, 3.54996793146960497053355363384e356,  0 },
	{ 193, 6.85143810773633759312975851330e358,  0 },
	{ 194, 1.32917899290084949306717315158e361,  0 },
	{ 195, 2.59189903615665651148098764559e363,  0 },
	{ 196, 5.08012211086704676250273578535e365,  0 },
	{ 197, 1.00078405584080821221303894971e368,  0 },
	{ 198, 1.98155243056480026018181712043e370,  0 },
	{ 199, 3.94328933682395251776181606966e372,  0 },
	{ 200, 7.88657867364790503552363213932e374,  0 }
	*/
};

int gsl_sf_lngamma_e(double x, gsl_sf_result * result);

int gsl_sf_lnfact_e(const unsigned int n, gsl_sf_result * result)
{
	/* CHECK_POINTER(result) */

	if (n <= GSL_SF_FACT_NMAX) {
		result->val = log(fact_table[n].f);
		result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
		return GSL_SUCCESS;
	}
	else {
		gsl_sf_lngamma_e(n + 1.0, result);
		return GSL_SUCCESS;
	}
}


#define PSI_1_TABLE_NMAX 100
__constant double psi_1_table[PSI_1_TABLE_NMAX + 1] = {
	0.0,
	M_PI*M_PI / 6.0,
	0.644934066848226436472415,
	0.394934066848226436472415,
	0.2838229557371153253613041,
	0.2213229557371153253613041,
	0.1813229557371153253613041,
	0.1535451779593375475835263,
	0.1331370146940314251345467,
	0.1175120146940314251345467,
	0.1051663356816857461222010,
	0.0951663356816857461222010,
	0.0869018728717683907503002,
	0.0799574284273239463058557,
	0.0740402686640103368384001,
	0.0689382278476838062261552,
	0.0644937834032393617817108,
	0.0605875334032393617817108,
	0.0571273257907826143768665,
	0.0540409060376961946237801,
	0.0512708229352031198315363,
	0.0487708229352031198315363,
	0.0465032492390579951149830,
	0.0444371335365786562720078,
	0.0425467743683366902984728,
	0.0408106632572255791873617,
	0.0392106632572255791873617,
	0.0377313733163971768204978,
	0.0363596312039143235969038,
	0.0350841209998326909438426,
	0.0338950603577399442137594,
	0.0327839492466288331026483,
	0.0317433665203020901265817,
	0.03076680402030209012658168,
	0.02984853037475571730748159,
	0.02898347847164153045627052,
	0.02816715194102928555831133,
	0.02739554700275768062003973,
	0.02666508681283803124093089,
	0.02597256603721476254286995,
	0.02531510384129102815759710,
	0.02469010384129102815759710,
	0.02409521984367056414807896,
	0.02352832641963428296894063,
	0.02298749353699501850166102,
	0.02247096461137518379091722,
	0.02197713745088135663042339,
	0.02150454765882086513703965,
	0.02105185413233829383780923,
	0.02061782635456051606003145,
	0.02020133322669712580597065,
	0.01980133322669712580597065,
	0.01941686571420193164987683,
	0.01904704322899483105816086,
	0.01869104465298913508094477,
	0.01834810912486842177504628,
	0.01801753061247172756017024,
	0.01769865306145131939690494,
	0.01739086605006319997554452,
	0.01709360088954001329302371,
	0.01680632711763538818529605,
	0.01652854933985761040751827,
	0.01625980437882562975715546,
	0.01599965869724394401313881,
	0.01574770606433893015574400,
	0.01550356543933893015574400,
	0.01526687904880638577704578,
	0.01503731063741979257227076,
	0.01481454387422086185273411,
	0.01459828089844231513993134,
	0.01438824099085987447620523,
	0.01418415935820681325171544,
	0.01398578601958352422176106,
	0.01379288478501562298719316,
	0.01360523231738567365335942,
	0.01342261726990576130858221,
	0.01324483949212798353080444,
	0.01307170929822216635628920,
	0.01290304679189732236910755,
	0.01273868124291638877278934,
	0.01257845051066194236996928,
	0.01242220051066194236996928,
	0.01226978472038606978956995,
	0.01212106372098095378719041,
	0.01197590477193174490346273,
	0.01183418141592267460867815,
	0.01169577311142440471248438,
	0.01156056489076458859566448,
	0.01142844704164317229232189,
	0.01129931481023821361463594,
	0.01117306812421372175754719,
	0.01104961133409026496742374,
	0.01092885297157366069257770,
	0.01081070552355853781923177,
	0.01069508522063334415522437,
	0.01058191183901270133041676,
	0.01047110851491297833872701,
	0.01036260157046853389428257,
	0.01025632035036012704977199,
	0.01015219706839427948625679,
	0.01005016666333357139524567
};


int gsl_sf_psi_1_int_e(const int n, gsl_sf_result * result)
{
	/* CHECK_POINTER(result) */
	if (n <= 0) {
		DOMAIN_ERROR(result);
	}
	else if (n <= PSI_1_TABLE_NMAX) {
		result->val = psi_1_table[n];
		result->err = GSL_DBL_EPSILON * result->val;
		return GSL_SUCCESS;
	}
	else {
		/* Abramowitz+Stegun 6.4.12
		* double-precision for n > 100
		*/
		double c0 = -1.0 / 30.0;
		double c1 = 1.0 / 42.0;
		double c2 = -1.0 / 30.0;
		double ni2 = (1.0 / n)*(1.0 / n);
		double ser = ni2*ni2 * (c0 + ni2*(c1 + c2*ni2));
		result->val = (1.0 + 0.5 / n + 1.0 / (6.0*n*n) + ser) / n;
		result->err = GSL_DBL_EPSILON * result->val;
		return GSL_SUCCESS;
	}
}

/* data for a Chebyshev series over a given interval */

struct cheb_series_struct {
	__constant double * c;   /* coefficients                */
	int order;    /* order of expansion          */
	double a;     /* lower interval point        */
	double b;     /* upper interval point        */
	int order_sp; /* effective single precision order */
};
typedef struct cheb_series_struct cheb_series;

static inline int
cheb_eval_e(__constant cheb_series * cs,
	double x,
	gsl_sf_result * result)
{
	int j;
	double d = 0.0;
	double dd = 0.0;

	double y = (2.0*x - cs->a - cs->b) / (cs->b - cs->a);
	double y2 = 2.0 * y;

	double e = 0.0;

	for (j = cs->order; j >= 1; j--) {
		double temp = d;
		d = y2*d - dd + cs->c[j];
		e += fabs(y2*temp) + fabs(dd) + fabs(cs->c[j]);
		dd = temp;
	}

	{
		double temp = d;
		d = y*d - dd + 0.5 * cs->c[0];
		e += fabs(y*temp) + fabs(dd) + 0.5 * fabs(cs->c[0]);
	}

	result->val = d;
	result->err = GSL_DBL_EPSILON * e + fabs(cs->c[cs->order]);

	return GSL_SUCCESS;
}

__constant double apsics_data[16] = {
	-.0204749044678185,
	-.0101801271534859,
	.0000559718725387,
	-.0000012917176570,
	.0000000572858606,
	-.0000000038213539,
	.0000000003397434,
	-.0000000000374838,
	.0000000000048990,
	-.0000000000007344,
	.0000000000001233,
	-.0000000000000228,
	.0000000000000045,
	-.0000000000000009,
	.0000000000000002,
	-.0000000000000000
};
__constant cheb_series apsi_cs = {
	apsics_data,
	15,
	-1, 1,
	9
};

/* Chebyshev fits from SLATEC code for psi(x)

Series for PSI        on the interval  0.         to  1.00000D+00
with weighted error   2.03E-17
log weighted error  16.69
significant figures required  16.39
decimal places required  17.37

Series for APSI       on the interval  0.         to  2.50000D-01
with weighted error   5.54E-17
log weighted error  16.26
significant figures required  14.42
decimal places required  16.86

*/

__constant double psics_data[23] = {
	-.038057080835217922,
	.491415393029387130,
	-.056815747821244730,
	.008357821225914313,
	-.001333232857994342,
	.000220313287069308,
	-.000037040238178456,
	.000006283793654854,
	-.000001071263908506,
	.000000183128394654,
	-.000000031353509361,
	.000000005372808776,
	-.000000000921168141,
	.000000000157981265,
	-.000000000027098646,
	.000000000004648722,
	-.000000000000797527,
	.000000000000136827,
	-.000000000000023475,
	.000000000000004027,
	-.000000000000000691,
	.000000000000000118,
	-.000000000000000020
};
__constant cheb_series psi_cs = {
	psics_data,
	22,
	-1, 1,
	17
};


/* digamma for x both positive and negative; we do both
* cases here because of the way we use even/odd parts
* of the function
*/
static int
psi_x(double x, gsl_sf_result * result)
{
	double y = fabs(x);

	if (x == 0.0 || x == -1.0 || x == -2.0) {
		DOMAIN_ERROR(result);
	}
	else if (y >= 2.0) {
		double t = 8.0 / (y*y) - 1.0;
		gsl_sf_result result_c;
		cheb_eval_e(&apsi_cs, t, &result_c);
		if (x < 0.0) {
			double s = sin(M_PI*x);
			double c = cos(M_PI*x);
			if (fabs(s) < 2.0*GSL_SQRT_DBL_MIN) {
				DOMAIN_ERROR(result);
			}
			else {
				result->val = log(y) - 0.5 / x + result_c.val - M_PI * c / s;
				result->err = M_PI*fabs(x)*GSL_DBL_EPSILON / (s*s);
				result->err += result_c.err;
				result->err += GSL_DBL_EPSILON * fabs(result->val);
				return GSL_SUCCESS;
			}
		}
		else {
			result->val = log(y) - 0.5 / x + result_c.val;
			result->err = result_c.err;
			result->err += GSL_DBL_EPSILON * fabs(result->val);
			return GSL_SUCCESS;
		}
	}
	else { /* -2 < x < 2 */
		gsl_sf_result result_c;

		if (x < -1.0) { /* x = -2 + v */
			double v = x + 2.0;
			double t1 = 1.0 / x;
			double t2 = 1.0 / (x + 1.0);
			double t3 = 1.0 / v;
			cheb_eval_e(&psi_cs, 2.0*v - 1.0, &result_c);

			result->val = -(t1 + t2 + t3) + result_c.val;
			result->err = GSL_DBL_EPSILON * (fabs(t1) + fabs(x / (t2*t2)) + fabs(x / (t3*t3)));
			result->err += result_c.err;
			result->err += GSL_DBL_EPSILON * fabs(result->val);
			return GSL_SUCCESS;
		}
		else if (x < 0.0) { /* x = -1 + v */
			double v = x + 1.0;
			double t1 = 1.0 / x;
			double t2 = 1.0 / v;
			cheb_eval_e(&psi_cs, 2.0*v - 1.0, &result_c);

			result->val = -(t1 + t2) + result_c.val;
			result->err = GSL_DBL_EPSILON * (fabs(t1) + fabs(x / (t2*t2)));
			result->err += result_c.err;
			result->err += GSL_DBL_EPSILON * fabs(result->val);
			return GSL_SUCCESS;
		}
		else if (x < 1.0) { /* x = v */
			double t1 = 1.0 / x;
			cheb_eval_e(&psi_cs, 2.0*x - 1.0, &result_c);

			result->val = -t1 + result_c.val;
			result->err = GSL_DBL_EPSILON * t1;
			result->err += result_c.err;
			result->err += GSL_DBL_EPSILON * fabs(result->val);
			return GSL_SUCCESS;
		}
		else { /* x = 1 + v */
			double v = x - 1.0;
			return cheb_eval_e(&psi_cs, 2.0*v - 1.0, result);
		}
	}
}

int gsl_sf_psi_e(double x, gsl_sf_result * result)
{
	/* CHECK_POINTER(result) */
	return psi_x(x, result);
}

/* coefficients for Maclaurin summation in hzeta()
* B_{2j}/(2j)!
*/
__constant double hzeta_c[15] = {
	1.00000000000000000000000000000,
	0.083333333333333333333333333333,
	-0.00138888888888888888888888888889,
	0.000033068783068783068783068783069,
	-8.2671957671957671957671957672e-07,
	2.0876756987868098979210090321e-08,
	-5.2841901386874931848476822022e-10,
	1.3382536530684678832826980975e-11,
	-3.3896802963225828668301953912e-13,
	8.5860620562778445641359054504e-15,
	-2.1748686985580618730415164239e-16,
	5.5090028283602295152026526089e-18,
	-1.3954464685812523340707686264e-19,
	3.5347070396294674716932299778e-21,
	-8.9535174270375468504026113181e-23
};

int gsl_sf_hzeta_e(double s, double q, gsl_sf_result * result)
{
	/* CHECK_POINTER(result) */

	if (s <= 1.0 || q <= 0.0) {
		DOMAIN_ERROR(result);
	}
	else {
		double max_bits = 54.0;
		double ln_term0 = -s * log(q);

		if (ln_term0 < GSL_LOG_DBL_MIN + 1.0) {
			UNDERFLOW_ERROR(result);
		}
		else if (ln_term0 > GSL_LOG_DBL_MAX - 1.0) {
			OVERFLOW_ERROR(result);
		}
		else if ((s > max_bits && q < 1.0) || (s > 0.5*max_bits && q < 0.25)) {
			result->val = pow(q, -s);
			result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
			return GSL_SUCCESS;
		}
		else if (s > 0.5*max_bits && q < 1.0) {
			double p1 = pow(q, -s);
			double p2 = pow(q / (1.0 + q), s);
			double p3 = pow(q / (2.0 + q), s);
			result->val = p1 * (1.0 + p2 + p3);
			result->err = GSL_DBL_EPSILON * (0.5*s + 2.0) * fabs(result->val);
			return GSL_SUCCESS;
		}
		else {
			/* Euler-Maclaurin summation formula
			* [Moshier, p. 400, with several typo corrections]
			*/
			int jmax = 12;
			int kmax = 10;
			int j, k;
			double pmax = pow(kmax + q, -s);
			double scp = s;
			double pcp = pmax / (kmax + q);
			double ans = pmax*((kmax + q) / (s - 1.0) + 0.5);

			for (k = 0; k<kmax; k++) {
				ans += pow(k + q, -s);
			}

			for (j = 0; j <= jmax; j++) {
				double delta = hzeta_c[j + 1] * scp * pcp;
				ans += delta;
				if (fabs(delta / ans) < 0.5*GSL_DBL_EPSILON) break;
				scp *= (s + 2 * j + 1)*(s + 2 * j + 2);
				pcp /= (kmax + q)*(kmax + q);
			}

			result->val = ans;
			result->err = 2.0 * (jmax + 1.0) * GSL_DBL_EPSILON * fabs(ans);
			return GSL_SUCCESS;
		}
	}
}

/* Chebyshev expansion for (log(1 + x(t)) - x(t))/x(t)^2
*
* x(t) = (4t-1)/(2(4-t))
* t(x) = (8x+1)/(2(x+2))
* -1/2 < x < 1/2
* -1 < t < 1
*/
__constant double lopxmx_data[20] = {
	-1.12100231323744103373737274541,
	0.19553462773379386241549597019,
	-0.01467470453808083971825344956,
	0.00166678250474365477643629067,
	-0.00018543356147700369785746902,
	0.00002280154021771635036301071,
	-2.8031253116633521699214134172e-06,
	3.5936568872522162983669541401e-07,
	-4.6241857041062060284381167925e-08,
	6.0822637459403991012451054971e-09,
	-8.0339824424815790302621320732e-10,
	1.0751718277499375044851551587e-10,
	-1.4445310914224613448759230882e-11,
	1.9573912180610336168921438426e-12,
	-2.6614436796793061741564104510e-13,
	3.6402634315269586532158344584e-14,
	-4.9937495922755006545809120531e-15,
	6.8802890218846809524646902703e-16,
	-9.5034129794804273611403251480e-17,
	1.3170135013050997157326965813e-17
};
__constant cheb_series lopxmx_cs = {
	lopxmx_data,
	19,
	-1, 1,
	9
};

int
gsl_sf_log_1plusx_mx_e(double x, gsl_sf_result * result)
{
	/* CHECK_POINTER(result) */

	if (x <= -1.0) {
		DOMAIN_ERROR(result);
	}
	else if (fabs(x) < GSL_ROOT5_DBL_EPSILON) {
		double c1 = -0.5;
		double c2 = 1.0 / 3.0;
		double c3 = -1.0 / 4.0;
		double c4 = 1.0 / 5.0;
		double c5 = -1.0 / 6.0;
		double c6 = 1.0 / 7.0;
		double c7 = -1.0 / 8.0;
		double c8 = 1.0 / 9.0;
		double c9 = -1.0 / 10.0;
		double t = c5 + x*(c6 + x*(c7 + x*(c8 + x*c9)));
		result->val = x*x * (c1 + x*(c2 + x*(c3 + x*(c4 + x*t))));
		result->err = GSL_DBL_EPSILON * fabs(result->val);
		return GSL_SUCCESS;
	}
	else if (fabs(x) < 0.5) {
		double t = 0.5*(8.0*x + 1.0) / (x + 2.0);
		gsl_sf_result c;
		cheb_eval_e(&lopxmx_cs, t, &c);
		result->val = x*x * c.val;
		result->err = x*x * c.err;
		return GSL_SUCCESS;
	}
	else {
		double lterm = log(1.0 + x);
		result->val = lterm - x;
		result->err = GSL_DBL_EPSILON * (fabs(lterm) + fabs(x));
		return GSL_SUCCESS;
	}
}

/* generic polygamma; assumes n >= 0 and x > 0
*/
static int
psi_n_xg0(int n, double x, gsl_sf_result * result)
{
	if (n == 0) {
		return gsl_sf_psi_e(x, result);
	}
	else {
		/* Abramowitz + Stegun 6.4.10 */
		gsl_sf_result ln_nf;
		gsl_sf_result hzeta;
		int stat_hz = gsl_sf_hzeta_e(n + 1.0, x, &hzeta);
		int stat_nf = gsl_sf_lnfact_e((unsigned int)n, &ln_nf);
		int stat_e = gsl_sf_exp_mult_err_e(ln_nf.val, ln_nf.err,
			hzeta.val, hzeta.err,
			result);
		if (GSL_IS_EVEN(n)) result->val = -result->val;
		return GSL_ERROR_SELECT_3(stat_e, stat_nf, stat_hz);
	}
}

int gsl_sf_psi_1_e(double x, gsl_sf_result * result)
{
	/* CHECK_POINTER(result) */

	if (x == 0.0 || x == -1.0 || x == -2.0) {
		DOMAIN_ERROR(result);
	}
	else if (x > 0.0)
	{
		return psi_n_xg0(1, x, result);
	}
	else if (x > -5.0)
	{
		/* Abramowitz + Stegun 6.4.6 */
		int M = -floor(x);
		double fx = x + M;
		double sum = 0.0;
		int m;

		if (fx == 0.0)
			DOMAIN_ERROR(result);

		for (m = 0; m < M; ++m)
			sum += 1.0 / ((x + m)*(x + m));

		{
			int stat_psi = psi_n_xg0(1, fx, result);
			result->val += sum;
			result->err += M * GSL_DBL_EPSILON * sum;
			return stat_psi;
		}
	}
	else
	{
		/* Abramowitz + Stegun 6.4.7 */
		double sin_px = sin(M_PI * x);
		double d = M_PI*M_PI / (sin_px*sin_px);
		gsl_sf_result r;
		int stat_psi = psi_n_xg0(1, 1.0 - x, &r);
		result->val = d - r.val;
		result->err = r.err + 2.0*GSL_DBL_EPSILON*d;
		return stat_psi;
	}
}

int gsl_sf_psi_n_e(int n, double x, gsl_sf_result * result)
{
	/* CHECK_POINTER(result) */

	if (n == 0)
	{
		return gsl_sf_psi_e(x, result);
	}
	else if (n == 1)
	{
		return gsl_sf_psi_1_e(x, result);
	}
	else if (n < 0 || x <= 0.0) {
		DOMAIN_ERROR(result);
	}
	else {
		gsl_sf_result ln_nf;
		gsl_sf_result hzeta;
		int stat_hz = gsl_sf_hzeta_e(n + 1.0, x, &hzeta);
		int stat_nf = gsl_sf_lnfact_e((unsigned int)n, &ln_nf);
		int stat_e = gsl_sf_exp_mult_err_e(ln_nf.val, ln_nf.err,
			hzeta.val, hzeta.err,
			result);
		if (GSL_IS_EVEN(n)) result->val = -result->val;
		return GSL_ERROR_SELECT_3(stat_e, stat_nf, stat_hz);
	}
}


/* x near a negative integer
* Calculates sign as well as log(|gamma(x)|).
* x = -N + eps
* assumes N >= 1
*/
static
int
lngamma_sgn_sing(int N, double eps, gsl_sf_result * lng, double * sgn)
{
	if (eps == 0.0) {
		lng->val = 0.0;
		lng->err = 0.0;
		*sgn = 0.0;
		GSL_ERROR(GSL_EDOM);
	}
	else if (N == 1) {
		/* calculate series for
		* g = eps gamma(-1+eps) + 1 + eps/2 (1+3eps)/(1-eps^2)
		* double-precision for |eps| < 0.02
		*/
		double c0 = 0.07721566490153286061;
		double c1 = 0.08815966957356030521;
		double c2 = -0.00436125434555340577;
		double c3 = 0.01391065882004640689;
		double c4 = -0.00409427227680839100;
		double c5 = 0.00275661310191541584;
		double c6 = -0.00124162645565305019;
		double c7 = 0.00065267976121802783;
		double c8 = -0.00032205261682710437;
		double c9 = 0.00016229131039545456;
		double g5 = c5 + eps*(c6 + eps*(c7 + eps*(c8 + eps*c9)));
		double g = eps*(c0 + eps*(c1 + eps*(c2 + eps*(c3 + eps*(c4 + eps*g5)))));

		/* calculate eps gamma(-1+eps), a negative quantity */
		double gam_e = g - 1.0 - 0.5*eps*(1.0 + 3.0*eps) / (1.0 - eps*eps);

		lng->val = log(fabs(gam_e) / fabs(eps));
		lng->err = 2.0 * GSL_DBL_EPSILON * fabs(lng->val);
		*sgn = (eps > 0.0 ? -1.0 : 1.0);
		return GSL_SUCCESS;
	}
	else {
		double g;

		/* series for sin(Pi(N+1-eps))/(Pi eps) modulo the sign
		* double-precision for |eps| < 0.02
		*/
		double cs1 = -1.6449340668482264365;
		double cs2 = 0.8117424252833536436;
		double cs3 = -0.1907518241220842137;
		double cs4 = 0.0261478478176548005;
		double cs5 = -0.0023460810354558236;
		double e2 = eps*eps;
		double sin_ser = 1.0 + e2*(cs1 + e2*(cs2 + e2*(cs3 + e2*(cs4 + e2*cs5))));

		/* calculate series for ln(gamma(1+N-eps))
		* double-precision for |eps| < 0.02
		*/
		double aeps = fabs(eps);
		double c1, c2, c3, c4, c5, c6, c7;
		double lng_ser;
		gsl_sf_result c0;
		gsl_sf_result psi_0;
		gsl_sf_result psi_1;
		gsl_sf_result psi_2;
		gsl_sf_result psi_3;
		gsl_sf_result psi_4;
		gsl_sf_result psi_5;
		gsl_sf_result psi_6;
		psi_2.val = 0.0;
		psi_3.val = 0.0;
		psi_4.val = 0.0;
		psi_5.val = 0.0;
		psi_6.val = 0.0;
		gsl_sf_lnfact_e(N, &c0);
		gsl_sf_psi_int_e(N + 1, &psi_0);
		gsl_sf_psi_1_int_e(N + 1, &psi_1);
		if (aeps > 0.00001) gsl_sf_psi_n_e(2, N + 1.0, &psi_2);
		if (aeps > 0.0002)  gsl_sf_psi_n_e(3, N + 1.0, &psi_3);
		if (aeps > 0.001)   gsl_sf_psi_n_e(4, N + 1.0, &psi_4);
		if (aeps > 0.005)   gsl_sf_psi_n_e(5, N + 1.0, &psi_5);
		if (aeps > 0.01)    gsl_sf_psi_n_e(6, N + 1.0, &psi_6);
		c1 = psi_0.val;
		c2 = psi_1.val / 2.0;
		c3 = psi_2.val / 6.0;
		c4 = psi_3.val / 24.0;
		c5 = psi_4.val / 120.0;
		c6 = psi_5.val / 720.0;
		c7 = psi_6.val / 5040.0;
		lng_ser = c0.val - eps*(c1 - eps*(c2 - eps*(c3 - eps*(c4 - eps*(c5 - eps*(c6 - eps*c7))))));

		/* calculate
		* g = ln(|eps gamma(-N+eps)|)
		*   = -ln(gamma(1+N-eps)) + ln(|eps Pi/sin(Pi(N+1+eps))|)
		*/
		g = -lng_ser - log(sin_ser);

		lng->val = g - log(fabs(eps));
		lng->err = c0.err + 2.0 * GSL_DBL_EPSILON * (fabs(g) + fabs(lng->val));

		*sgn = (GSL_IS_ODD(N) ? -1.0 : 1.0) * (eps > 0.0 ? 1.0 : -1.0);

		return GSL_SUCCESS;
	}
}

inline
static
int
lngamma_1_pade(double eps, gsl_sf_result * result)
{
	/* Use (2,2) Pade for Log[Gamma[1+eps]]/eps
	* plus a correction series.
	*/
	double n1 = -1.0017419282349508699871138440;
	double n2 = 1.7364839209922879823280541733;
	double d1 = 1.2433006018858751556055436011;
	double d2 = 5.0456274100274010152489597514;
	double num = (eps + n1) * (eps + n2);
	double den = (eps + d1) * (eps + d2);
	double pade = 2.0816265188662692474880210318 * num / den;
	double c0 = 0.004785324257581753;
	double c1 = -0.01192457083645441;
	double c2 = 0.01931961413960498;
	double c3 = -0.02594027398725020;
	double c4 = 0.03141928755021455;
	double eps5 = eps*eps*eps*eps*eps;
	double corr = eps5 * (c0 + eps*(c1 + eps*(c2 + eps*(c3 + c4*eps))));
	result->val = eps * (pade + corr);
	result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
	return GSL_SUCCESS;
}

inline
static
int
lngamma_2_pade(double eps, gsl_sf_result * result)
{
	/* Use (2,2) Pade for Log[Gamma[2+eps]]/eps
	* plus a correction series.
	*/
	double n1 = 1.000895834786669227164446568;
	double n2 = 4.209376735287755081642901277;
	double d1 = 2.618851904903217274682578255;
	double d2 = 10.85766559900983515322922936;
	double num = (eps + n1) * (eps + n2);
	double den = (eps + d1) * (eps + d2);
	double pade = 2.85337998765781918463568869 * num / den;
	double c0 = 0.0001139406357036744;
	double c1 = -0.0001365435269792533;
	double c2 = 0.0001067287169183665;
	double c3 = -0.0000693271800931282;
	double c4 = 0.0000407220927867950;
	double eps5 = eps*eps*eps*eps*eps;
	double corr = eps5 * (c0 + eps*(c1 + eps*(c2 + eps*(c3 + c4*eps))));
	result->val = eps * (pade + corr);
	result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
	return GSL_SUCCESS;
}

/* x = eps near zero
* gives double-precision for |eps| < 0.02
*/
static
int
lngamma_sgn_0(double eps, gsl_sf_result * lng, double * sgn)
{
	/* calculate series for g(eps) = Gamma(eps) eps - 1/(1+eps) - eps/2 */
	double c1 = -0.07721566490153286061;
	double c2 = -0.01094400467202744461;
	double c3 = 0.09252092391911371098;
	double c4 = -0.01827191316559981266;
	double c5 = 0.01800493109685479790;
	double c6 = -0.00685088537872380685;
	double c7 = 0.00399823955756846603;
	double c8 = -0.00189430621687107802;
	double c9 = 0.00097473237804513221;
	double c10 = -0.00048434392722255893;
	double g6 = c6 + eps*(c7 + eps*(c8 + eps*(c9 + eps*c10)));
	double g = eps*(c1 + eps*(c2 + eps*(c3 + eps*(c4 + eps*(c5 + eps*g6)))));

	/* calculate Gamma(eps) eps, a positive quantity */
	double gee = g + 1.0 / (1.0 + eps) + 0.5*eps;

	lng->val = log(gee / fabs(eps));
	lng->err = 4.0 * GSL_DBL_EPSILON * fabs(lng->val);
	*sgn = GSL_SIGN(eps);

	return GSL_SUCCESS;
}

/* coefficients for gamma=7, kmax=8  Lanczos method */
__constant double lanczos_7_c[9] = {
	0.99999999999980993227684700473478,
	676.520368121885098567009190444019,
	-1259.13921672240287047156078755283,
	771.3234287776530788486528258894,
	-176.61502916214059906584551354,
	12.507343278686904814458936853,
	-0.13857109526572011689554707,
	9.984369578019570859563e-6,
	1.50563273514931155834e-7
};


/* Lanczos method for real x > 0;
* gamma=7, truncated at 1/(z+8)
* [J. SIAM Numer. Anal, Ser. B, 1 (1964) 86]
*/
static
int
lngamma_lanczos(double x, gsl_sf_result * result)
{
	int k;
	double Ag;
	double term1, term2;

	x -= 1.0; /* Lanczos writes z! instead of Gamma(z) */

	Ag = lanczos_7_c[0];
	for (k = 1; k <= 8; k++) { Ag += lanczos_7_c[k] / (x + k); }

	/* (x+0.5)*log(x+7.5) - (x+7.5) + LogRootTwoPi_ + log(Ag(x)) */
	term1 = (x + 0.5)*log((x + 7.5) / M_E);
	term2 = LogRootTwoPi_ + log(Ag);
	result->val = term1 + (term2 - 7.0);
	result->err = 2.0 * GSL_DBL_EPSILON * (fabs(term1) + fabs(term2) + 7.0);
	result->err += GSL_DBL_EPSILON * fabs(result->val);

	return GSL_SUCCESS;
}

int gsl_sf_lngamma_e(double x, gsl_sf_result * result)
{
	/* CHECK_POINTER(result) */

	if (fabs(x - 1.0) < 0.01) {
		/* Note that we must amplify the errors
		* from the Pade evaluations because of
		* the way we must pass the argument, i.e.
		* writing (1-x) is a loss of precision
		* when x is near 1.
		*/
		int stat = lngamma_1_pade(x - 1.0, result);
		result->err *= 1.0 / (GSL_DBL_EPSILON + fabs(x - 1.0));
		return stat;
	}
	else if (fabs(x - 2.0) < 0.01) {
		int stat = lngamma_2_pade(x - 2.0, result);
		result->err *= 1.0 / (GSL_DBL_EPSILON + fabs(x - 2.0));
		return stat;
	}
	else if (x >= 0.5) {
		return lngamma_lanczos(x, result);
	}
	else if (x == 0.0) {
		DOMAIN_ERROR(result);
	}
	else if (fabs(x) < 0.02) {
		double sgn;
		return lngamma_sgn_0(x, result, &sgn);
	}
	else if (x > -0.5 / (GSL_DBL_EPSILON*M_PI)) {
		/* Try to extract a fractional
		* part from x.
		*/
		double z = 1.0 - x;
		double s = sin(M_PI*z);
		double as = fabs(s);
		if (s == 0.0) {
			DOMAIN_ERROR(result);
		}
		else if (as < M_PI*0.015) {
			/* x is near a negative integer, -N */
			if (x < INT_MIN + 2.0) {
				result->val = 0.0;
				result->err = 0.0;
				GSL_ERROR(GSL_EROUND);
			}
			else {
				int N = -(int)(x - 0.5);
				double eps = x + N;
				double sgn;
				return lngamma_sgn_sing(N, eps, result, &sgn);
			}
		}
		else {
			gsl_sf_result lg_z;
			lngamma_lanczos(z, &lg_z);
			result->val = M_LNPI - (log(as) + lg_z.val);
			result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val) + lg_z.err;
			return GSL_SUCCESS;
		}
	}
	else {
		/* |x| was too large to extract any fractional part */
		result->val = 0.0;
		result->err = 0.0;
		GSL_ERROR(GSL_EROUND);
	}
}

int
gsl_sf_exp_err_e(double x, double dx, gsl_sf_result * result)
{
	double adx = fabs(dx);

	/* CHECK_POINTER(result) */

	if (x + adx > GSL_LOG_DBL_MAX) {
		OVERFLOW_ERROR(result);
	}
	else if (x - adx < GSL_LOG_DBL_MIN) {
		UNDERFLOW_ERROR(result);
	}
	else {
		double ex = exp(x);
		double edx = exp(adx);
		result->val = ex;
		result->err = ex * GSL_MAX_DBL(GSL_DBL_EPSILON, edx - 1.0 / edx);
		result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
		return GSL_SUCCESS;
	}
}

/* Chebyshev coefficients for Gamma*(3/4(t+1)+1/2), -1<t<1
*/
__constant double gstar_a_data[30] = {
	2.16786447866463034423060819465,
	-0.05533249018745584258035832802,
	0.01800392431460719960888319748,
	-0.00580919269468937714480019814,
	0.00186523689488400339978881560,
	-0.00059746524113955531852595159,
	0.00019125169907783353925426722,
	-0.00006124996546944685735909697,
	0.00001963889633130842586440945,
	-6.3067741254637180272515795142e-06,
	2.0288698405861392526872789863e-06,
	-6.5384896660838465981983750582e-07,
	2.1108698058908865476480734911e-07,
	-6.8260714912274941677892994580e-08,
	2.2108560875880560555583978510e-08,
	-7.1710331930255456643627187187e-09,
	2.3290892983985406754602564745e-09,
	-7.5740371598505586754890405359e-10,
	2.4658267222594334398525312084e-10,
	-8.0362243171659883803428749516e-11,
	2.6215616826341594653521346229e-11,
	-8.5596155025948750540420068109e-12,
	2.7970831499487963614315315444e-12,
	-9.1471771211886202805502562414e-13,
	2.9934720198063397094916415927e-13,
	-9.8026575909753445931073620469e-14,
	3.2116773667767153777571410671e-14,
	-1.0518035333878147029650507254e-14,
	3.4144405720185253938994854173e-15,
	-1.0115153943081187052322643819e-15
};
__constant cheb_series gstar_a_cs = {
	gstar_a_data,
	29,
	-1, 1,
	17
};


/* Chebyshev coefficients for
* x^2(Gamma*(x) - 1 - 1/(12x)), x = 4(t+1)+2, -1 < t < 1
*/
__constant double gstar_b_data[] = {
	0.0057502277273114339831606096782,
	0.0004496689534965685038254147807,
	-0.0001672763153188717308905047405,
	0.0000615137014913154794776670946,
	-0.0000223726551711525016380862195,
	8.0507405356647954540694800545e-06,
	-2.8671077107583395569766746448e-06,
	1.0106727053742747568362254106e-06,
	-3.5265558477595061262310873482e-07,
	1.2179216046419401193247254591e-07,
	-4.1619640180795366971160162267e-08,
	1.4066283500795206892487241294e-08,
	-4.6982570380537099016106141654e-09,
	1.5491248664620612686423108936e-09,
	-5.0340936319394885789686867772e-10,
	1.6084448673736032249959475006e-10,
	-5.0349733196835456497619787559e-11,
	1.5357154939762136997591808461e-11,
	-4.5233809655775649997667176224e-12,
	1.2664429179254447281068538964e-12,
	-3.2648287937449326771785041692e-13,
	7.1528272726086133795579071407e-14,
	-9.4831735252566034505739531258e-15,
	-2.3124001991413207293120906691e-15,
	2.8406613277170391482590129474e-15,
	-1.7245370321618816421281770927e-15,
	8.6507923128671112154695006592e-16,
	-3.9506563665427555895391869919e-16,
	1.6779342132074761078792361165e-16,
	-6.0483153034414765129837716260e-17
};
__constant cheb_series gstar_b_cs = {
	gstar_b_data,
	29,
	-1, 1,
	18
};


/* series for gammastar(x)
* double-precision for x > 10.0
*/
static
int
gammastar_ser(double x, gsl_sf_result * result)
{
	/* Use the Stirling series for the correction to Log(Gamma(x)),
	* which is better behaved and easier to compute than the
	* regular Stirling series for Gamma(x).
	*/
	double y = 1.0 / (x*x);
	double c0 = 1.0 / 12.0;
	double c1 = -1.0 / 360.0;
	double c2 = 1.0 / 1260.0;
	double c3 = -1.0 / 1680.0;
	double c4 = 1.0 / 1188.0;
	double c5 = -691.0 / 360360.0;
	double c6 = 1.0 / 156.0;
	double c7 = -3617.0 / 122400.0;
	double ser = c0 + y*(c1 + y*(c2 + y*(c3 + y*(c4 + y*(c5 + y*(c6 + y*c7))))));
	result->val = exp(ser / x);
	result->err = 2.0 * GSL_DBL_EPSILON * result->val * GSL_MAX_DBL(1.0, ser / x);
	return GSL_SUCCESS;
}

int
gsl_sf_gammastar_e(double x, gsl_sf_result * result)
{
	/* CHECK_POINTER(result) */

	if (x <= 0.0) {
		DOMAIN_ERROR(result);
	}
	else if (x < 0.5) {
		gsl_sf_result lg;
		int stat_lg = gsl_sf_lngamma_e(x, &lg);
		double lx = log(x);
		double c = 0.5*(M_LN2 + M_LNPI);
		double lnr_val = lg.val - (x - 0.5)*lx + x - c;
		double lnr_err = lg.err + 2.0 * GSL_DBL_EPSILON *((x + 0.5)*fabs(lx) + c);
		int stat_e = gsl_sf_exp_err_e(lnr_val, lnr_err, result);
		return GSL_ERROR_SELECT_2(stat_lg, stat_e);
	}
	else if (x < 2.0) {
		double t = 4.0 / 3.0*(x - 0.5) - 1.0;
		return cheb_eval_e(&gstar_a_cs, t, result);
	}
	else if (x < 10.0) {
		double t = 0.25*(x - 2.0) - 1.0;
		gsl_sf_result c;
		cheb_eval_e(&gstar_b_cs, t, &c);
		result->val = c.val / (x*x) + 1.0 + 1.0 / (12.0*x);
		result->err = c.err / (x*x);
		result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
		return GSL_SUCCESS;
	}
	else if (x < 1.0 / GSL_ROOT4_DBL_EPSILON) {
		return gammastar_ser(x, result);
	}
	else if (x < 1.0 / GSL_DBL_EPSILON) {
		/* Use Stirling formula for Gamma(x).
		*/
		double xi = 1.0 / x;
		result->val = 1.0 + xi / 12.0*(1.0 + xi / 24.0*(1.0 - xi*(139.0 / 180.0 + 571.0 / 8640.0*xi)));
		result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
		return GSL_SUCCESS;
	}
	else {
		result->val = 1.0;
		result->err = 1.0 / x;
		return GSL_SUCCESS;
	}
}

/* The dominant part,
* D(a,x) := x^a e^(-x) / Gamma(a+1)
*/
static
int
gamma_inc_D(double a, double x, gsl_sf_result * result)
{
	if (a < 10.0) {
		double lnr;
		gsl_sf_result lg;
		gsl_sf_lngamma_e(a + 1.0, &lg);
		lnr = a * log(x) - x - lg.val;
		result->val = exp(lnr);
		result->err = 2.0 * GSL_DBL_EPSILON * (fabs(lnr) + 1.0) * fabs(result->val);
		return GSL_SUCCESS;
	}
	else {
		gsl_sf_result gstar;
		gsl_sf_result ln_term;
		double term1;
		if (x < 0.5*a) {
			double u = x / a;
			double ln_u = log(u);
			ln_term.val = ln_u - u + 1.0;
			ln_term.err = (fabs(ln_u) + fabs(u) + 1.0) * GSL_DBL_EPSILON;
		}
		else {
			double mu = (x - a) / a;
			gsl_sf_log_1plusx_mx_e(mu, &ln_term);  /* log(1+mu) - mu */
												   /* Propagate cancellation error from x-a, since the absolute
												   error of mu=x-a is DBL_EPSILON */
			ln_term.err += GSL_DBL_EPSILON * fabs(mu);
		};
		gsl_sf_gammastar_e(a, &gstar);
		term1 = exp(a*ln_term.val) / sqrt(2.0*M_PI*a);
		result->val = term1 / gstar.val;
		result->err = 2.0 * GSL_DBL_EPSILON * (fabs(a*ln_term.val) + 1.0) * fabs(result->val);
		/* Include propagated error from log term */
		result->err += fabs(a) * ln_term.err * fabs(result->val);
		result->err += gstar.err / fabs(gstar.val) * fabs(result->val);
		return GSL_SUCCESS;
	}

}

/* Evaluate the continued fraction for exprel.
* [Abramowitz+Stegun, 4.2.41]
*/
static
int
exprel_n_CF(double N, double x, gsl_sf_result * result)
{
	double RECUR_BIG = GSL_SQRT_DBL_MAX;
	int maxiter = 5000;
	int n = 1;
	double Anm2 = 1.0;
	double Bnm2 = 0.0;
	double Anm1 = 0.0;
	double Bnm1 = 1.0;
	double a1 = 1.0;
	double b1 = 1.0;
	double a2 = -x;
	double b2 = N + 1;
	double an, bn;

	double fn;

	double An = b1*Anm1 + a1*Anm2;   /* A1 */
	double Bn = b1*Bnm1 + a1*Bnm2;   /* B1 */

									 /* One explicit step, before we get to the main pattern. */
	n++;
	Anm2 = Anm1;
	Bnm2 = Bnm1;
	Anm1 = An;
	Bnm1 = Bn;
	An = b2*Anm1 + a2*Anm2;   /* A2 */
	Bn = b2*Bnm1 + a2*Bnm2;   /* B2 */

	fn = An / Bn;

	while (n < maxiter) {
		double old_fn;
		double del;
		n++;
		Anm2 = Anm1;
		Bnm2 = Bnm1;
		Anm1 = An;
		Bnm1 = Bn;
		an = (GSL_IS_ODD(n) ? ((n - 1) / 2)*x : -(N + (n / 2) - 1)*x);
		bn = N + n - 1;
		An = bn*Anm1 + an*Anm2;
		Bn = bn*Bnm1 + an*Bnm2;

		if (fabs(An) > RECUR_BIG || fabs(Bn) > RECUR_BIG) {
			An /= RECUR_BIG;
			Bn /= RECUR_BIG;
			Anm1 /= RECUR_BIG;
			Bnm1 /= RECUR_BIG;
			Anm2 /= RECUR_BIG;
			Bnm2 /= RECUR_BIG;
		}

		old_fn = fn;
		fn = An / Bn;
		del = old_fn / fn;

		if (fabs(del - 1.0) < 2.0*GSL_DBL_EPSILON) break;
	}

	result->val = fn;
	result->err = 4.0*(n + 1.0)*GSL_DBL_EPSILON*fabs(fn);

	if (n == maxiter)
		GSL_ERROR(GSL_EMAXITER);
	else
		return GSL_SUCCESS;
}

int
gsl_sf_exprel_n_CF_e(double N, double x, gsl_sf_result * result)
{
	return exprel_n_CF(N, x, result);
}

/* P series representation.
*/
static
int
gamma_inc_P_series(double a, double x, gsl_sf_result * result)
{
	int nmax = 10000;

	gsl_sf_result D;
	int stat_D = gamma_inc_D(a, x, &D);

	/* Approximating the terms of the series using Stirling's
	approximation gives t_n = (x/a)^n * exp(-n(n+1)/(2a)), so the
	convergence condition is n^2 / (2a) + (1-(x/a) + (1/2a)) n >>
	-log(GSL_DBL_EPS) if we want t_n < O(1e-16) t_0. The condition
	below detects cases where the minimum value of n is > 5000 */

	if (x > 0.995 * a && a > 1e5) { /* Difficult case: try continued fraction */
		gsl_sf_result cf_res;
		int status = gsl_sf_exprel_n_CF_e(a, x, &cf_res);
		result->val = D.val * cf_res.val;
		result->err = fabs(D.val * cf_res.err) + fabs(D.err * cf_res.val);
		return status;
	}

	/* Series would require excessive number of terms */

	if (x > (a + nmax)) {
		GSL_ERROR(GSL_EMAXITER);
	}

	/* Normal case: sum the series */

	{
		double sum = 1.0;
		double term = 1.0;
		double remainder;
		int n;

		/* Handle lower part of the series where t_n is increasing, |x| > a+n */

		int nlow = (x > a) ? (x - a) : 0;

		for (n = 1; n < nlow; n++) {
			term *= x / (a + n);
			sum += term;
		}

		/* Handle upper part of the series where t_n is decreasing, |x| < a+n */

		for (/* n = previous n */; n<nmax; n++) {
			term *= x / (a + n);
			sum += term;
			if (fabs(term / sum) < GSL_DBL_EPSILON) break;
		}

		/*  Estimate remainder of series ~ t_(n+1)/(1-x/(a+n+1)) */
		{
			double tnp1 = (x / (a + n)) * term;
			remainder = tnp1 / (1.0 - x / (a + n + 1.0));
		}

		result->val = D.val * sum;
		result->err = D.err * fabs(sum) + fabs(D.val * remainder);
		result->err += (1.0 + n) * GSL_DBL_EPSILON * fabs(result->val);

		if (n == nmax && fabs(remainder / sum) > GSL_SQRT_DBL_EPSILON)
			GSL_ERROR(GSL_EMAXITER);
		else
			return stat_D;
	}
}

/*-*-*-*-*-*-*-*-*-*-*-* Functions with Error Codes *-*-*-*-*-*-*-*-*-*-*-*/

/* Chebyshev fit for erfc((t+1)/2), -1 < t < 1
*/
__constant double erfc_xlt1_data[20] = {
	1.06073416421769980345174155056,
	-0.42582445804381043569204735291,
	0.04955262679620434040357683080,
	0.00449293488768382749558001242,
	-0.00129194104658496953494224761,
	-0.00001836389292149396270416979,
	0.00002211114704099526291538556,
	-5.23337485234257134673693179020e-7,
	-2.78184788833537885382530989578e-7,
	1.41158092748813114560316684249e-8,
	2.72571296330561699984539141865e-9,
	-2.06343904872070629406401492476e-10,
	-2.14273991996785367924201401812e-11,
	2.22990255539358204580285098119e-12,
	1.36250074650698280575807934155e-13,
	-1.95144010922293091898995913038e-14,
	-6.85627169231704599442806370690e-16,
	1.44506492869699938239521607493e-16,
	2.45935306460536488037576200030e-18,
	-9.29599561220523396007359328540e-19
};
__constant cheb_series erfc_xlt1_cs = {
	erfc_xlt1_data,
	19,
	-1, 1,
	12
};

/* Chebyshev fit for erfc(x) exp(x^2), 1 < x < 5, x = 2t + 3, -1 < t < 1
*/
__constant double erfc_x15_data[25] = {
	0.44045832024338111077637466616,
	-0.143958836762168335790826895326,
	0.044786499817939267247056666937,
	-0.013343124200271211203618353102,
	0.003824682739750469767692372556,
	-0.001058699227195126547306482530,
	0.000283859419210073742736310108,
	-0.000073906170662206760483959432,
	0.000018725312521489179015872934,
	-4.62530981164919445131297264430e-6,
	1.11558657244432857487884006422e-6,
	-2.63098662650834130067808832725e-7,
	6.07462122724551777372119408710e-8,
	-1.37460865539865444777251011793e-8,
	3.05157051905475145520096717210e-9,
	-6.65174789720310713757307724790e-10,
	1.42483346273207784489792999706e-10,
	-3.00141127395323902092018744545e-11,
	6.22171792645348091472914001250e-12,
	-1.26994639225668496876152836555e-12,
	2.55385883033257575402681845385e-13,
	-5.06258237507038698392265499770e-14,
	9.89705409478327321641264227110e-15,
	-1.90685978789192181051961024995e-15,
	3.50826648032737849245113757340e-16
};
__constant cheb_series erfc_x15_cs = {
	erfc_x15_data,
	24,
	-1, 1,
	16
};

/* Chebyshev fit for erfc(x) x exp(x^2), 5 < x < 10, x = (5t + 15)/2, -1 < t < 1
*/
__constant double erfc_x510_data[20] = {
	1.11684990123545698684297865808,
	0.003736240359381998520654927536,
	-0.000916623948045470238763619870,
	0.000199094325044940833965078819,
	-0.000040276384918650072591781859,
	7.76515264697061049477127605790e-6,
	-1.44464794206689070402099225301e-6,
	2.61311930343463958393485241947e-7,
	-4.61833026634844152345304095560e-8,
	8.00253111512943601598732144340e-9,
	-1.36291114862793031395712122089e-9,
	2.28570483090160869607683087722e-10,
	-3.78022521563251805044056974560e-11,
	6.17253683874528285729910462130e-12,
	-9.96019290955316888445830597430e-13,
	1.58953143706980770269506726000e-13,
	-2.51045971047162509999527428316e-14,
	3.92607828989125810013581287560e-15,
	-6.07970619384160374392535453420e-16,
	9.12600607264794717315507477670e-17
};
__constant cheb_series erfc_x510_cs = {
	erfc_x510_data,
	19,
	-1, 1,
	12
};

__constant double erfc8_sumP[] = {
	2.97886562639399288862,
	7.409740605964741794425,
	6.1602098531096305440906,
	5.019049726784267463450058,
	1.275366644729965952479585264,
	0.5641895835477550741253201704
};
__constant double erfc8_sumQ[] = {
	3.3690752069827527677,
	9.608965327192787870698,
	17.08144074746600431571095,
	12.0489519278551290360340491,
	9.396034016235054150430579648,
	2.260528520767326969591866945,
	1.0
};

__constant double erfc8_sum(double x)
{
	/* estimates erfc(x) valid for 8 < x < 100 */
	/* This is based on index 5725 in Hart et al */

	double num = 0.0, den = 0.0;
	int i;

	num = erfc8_sumP[5];
	for (i = 4; i >= 0; --i) {
		num = x*num + erfc8_sumP[i];
	}
	den = erfc8_sumQ[6];
	for (i = 5; i >= 0; --i) {
		den = x*den + erfc8_sumQ[i];
	}

	return num / den;
}

inline
static double erfc8(double x)
{
	double e;
	e = erfc8_sum(x);
	e *= exp(-x*x);
	return e;
}

int gsl_sf_erfc_e(double x, gsl_sf_result * result)
{
	double ax = fabs(x);
	double e_val, e_err;

	/* CHECK_POINTER(result) */

	if (ax <= 1.0) {
		double t = 2.0*ax - 1.0;
		gsl_sf_result c;
		cheb_eval_e(&erfc_xlt1_cs, t, &c);
		e_val = c.val;
		e_err = c.err;
	}
	else if (ax <= 5.0) {
		double ex2 = exp(-x*x);
		double t = 0.5*(ax - 3.0);
		gsl_sf_result c;
		cheb_eval_e(&erfc_x15_cs, t, &c);
		e_val = ex2 * c.val;
		e_err = ex2 * (c.err + 2.0*fabs(x)*GSL_DBL_EPSILON);
	}
	else if (ax < 10.0) {
		double exterm = exp(-x*x) / ax;
		double t = (2.0*ax - 15.0) / 5.0;
		gsl_sf_result c;
		cheb_eval_e(&erfc_x510_cs, t, &c);
		e_val = exterm * c.val;
		e_err = exterm * (c.err + 2.0*fabs(x)*GSL_DBL_EPSILON + GSL_DBL_EPSILON);
	}
	else {
		e_val = erfc8(ax);
		e_err = (x*x + 1.0) * GSL_DBL_EPSILON * fabs(e_val);
	}

	if (x < 0.0) {
		result->val = 2.0 - e_val;
		result->err = e_err;
		result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
	}
	else {
		result->val = e_val;
		result->err = e_err;
		result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
	}

	return GSL_SUCCESS;
}

/* Uniform asymptotic for x near a, a and x large.
* See [Temme, p. 285]
*/
static
int
gamma_inc_Q_asymp_unif(double a, double x, gsl_sf_result * result)
{
	double rta = sqrt(a);
	double eps = (x - a) / a;

	gsl_sf_result ln_term;
	int stat_ln = gsl_sf_log_1plusx_mx_e(eps, &ln_term);  /* log(1+eps) - eps */
	double eta = GSL_SIGN(eps) * sqrt(-2.0*ln_term.val);

	gsl_sf_result erfc;

	double R;
	double c0, c1;

	/* This used to say erfc(eta*M_SQRT2*rta), which is wrong.
	* The sqrt(2) is in the denominator. Oops.
	* Fixed: [GJ] Mon Nov 15 13:25:32 MST 2004
	*/
	gsl_sf_erfc_e(eta*rta / M_SQRT2, &erfc);

	if (fabs(eps) < GSL_ROOT5_DBL_EPSILON) {
		c0 = -1.0 / 3.0 + eps*(1.0 / 12.0 - eps*(23.0 / 540.0 - eps*(353.0 / 12960.0 - eps*589.0 / 30240.0)));
		c1 = -1.0 / 540.0 - eps / 288.0;
	}
	else {
		double rt_term = sqrt(-2.0 * ln_term.val / (eps*eps));
		double lam = x / a;
		c0 = (1.0 - 1.0 / rt_term) / eps;
		c1 = -(eta*eta*eta * (lam*lam + 10.0*lam + 1.0) - 12.0 * eps*eps*eps) / (12.0 * eta*eta*eta*eps*eps*eps);
	}

	R = exp(-0.5*a*eta*eta) / (M_SQRT2*M_SQRTPI*rta) * (c0 + c1 / a);

	result->val = 0.5 * erfc.val + R;
	result->err = GSL_DBL_EPSILON * fabs(R * 0.5 * a*eta*eta) + 0.5 * erfc.err;
	result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);

	return stat_ln;
}


/* Useful for small a and x. Handles the subtraction analytically.
*/
static
int
gamma_inc_Q_series(double a, double x, gsl_sf_result * result)
{
	double term1;  /* 1 - x^a/Gamma(a+1) */
	double sum;    /* 1 + (a+1)/(a+2)(-x)/2! + (a+1)/(a+3)(-x)^2/3! */
	int stat_sum;
	double term2;  /* a temporary variable used at the end */

	{
		/* Evaluate series for 1 - x^a/Gamma(a+1), small a
		*/
		double pg21 = -2.404113806319188570799476;  /* PolyGamma[2,1] */
		double lnx = log(x);
		double el = M_EULER + lnx;
		double c1 = -el;
		double c2 = M_PI*M_PI / 12.0 - 0.5*el*el;
		double c3 = el*(M_PI*M_PI / 12.0 - el*el / 6.0) + pg21 / 6.0;
		double c4 = -0.04166666666666666667
			* (-1.758243446661483480 + lnx)
			* (-0.764428657272716373 + lnx)
			* (0.723980571623507657 + lnx)
			* (4.107554191916823640 + lnx);
		double c5 = -0.0083333333333333333
			* (-2.06563396085715900 + lnx)
			* (-1.28459889470864700 + lnx)
			* (-0.27583535756454143 + lnx)
			* (1.33677371336239618 + lnx)
			* (5.17537282427561550 + lnx);
		double c6 = -0.0013888888888888889
			* (-2.30814336454783200 + lnx)
			* (-1.65846557706987300 + lnx)
			* (-0.88768082560020400 + lnx)
			* (0.17043847751371778 + lnx)
			* (1.92135970115863890 + lnx)
			* (6.22578557795474900 + lnx);
		double c7 = -0.00019841269841269841
			* (-2.5078657901291800 + lnx)
			* (-1.9478900888958200 + lnx)
			* (-1.3194837322612730 + lnx)
			* (-0.5281322700249279 + lnx)
			* (0.5913834939078759 + lnx)
			* (2.4876819633378140 + lnx)
			* (7.2648160783762400 + lnx);
		double c8 = -0.00002480158730158730
			* (-2.677341544966400 + lnx)
			* (-2.182810448271700 + lnx)
			* (-1.649350342277400 + lnx)
			* (-1.014099048290790 + lnx)
			* (-0.191366955370652 + lnx)
			* (0.995403817918724 + lnx)
			* (3.041323283529310 + lnx)
			* (8.295966556941250 + lnx);
		double c9 = -2.75573192239859e-6
			* (-2.8243487670469080 + lnx)
			* (-2.3798494322701120 + lnx)
			* (-1.9143674728689960 + lnx)
			* (-1.3814529102920370 + lnx)
			* (-0.7294312810261694 + lnx)
			* (0.1299079285269565 + lnx)
			* (1.3873333251885240 + lnx)
			* (3.5857258865210760 + lnx)
			* (9.3214237073814600 + lnx);
		double c10 = -2.75573192239859e-7
			* (-2.9540329644556910 + lnx)
			* (-2.5491366926991850 + lnx)
			* (-2.1348279229279880 + lnx)
			* (-1.6741881076349450 + lnx)
			* (-1.1325949616098420 + lnx)
			* (-0.4590034650618494 + lnx)
			* (0.4399352987435699 + lnx)
			* (1.7702236517651670 + lnx)
			* (4.1231539047474080 + lnx)
			* (10.342627908148680 + lnx);

		term1 = a*(c1 + a*(c2 + a*(c3 + a*(c4 + a*(c5 + a*(c6 + a*(c7 + a*(c8 + a*(c9 + a*c10)))))))));
	}

	{
		/* Evaluate the sum.
		*/
		int nmax = 5000;
		double t = 1.0;
		int n;
		sum = 1.0;

		for (n = 1; n<nmax; n++) {
			t *= -x / (n + 1.0);
			sum += (a + 1.0) / (a + n + 1.0)*t;
			if (fabs(t / sum) < GSL_DBL_EPSILON) break;
		}

		if (n == nmax)
			stat_sum = GSL_EMAXITER;
		else
			stat_sum = GSL_SUCCESS;
	}

	term2 = (1.0 - term1) * a / (a + 1.0) * x * sum;
	result->val = term1 + term2;
	result->err = GSL_DBL_EPSILON * (fabs(term1) + 2.0*fabs(term2));
	result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
	return stat_sum;
}

/* Continued fraction for Q.
*
* Q(a,x) = D(a,x) a/x F(a,x)
*
* Hans E. Plesser, 2002-01-22 (hans dot plesser at itf dot nlh dot no):
*
* Since the Gautschi equivalent series method for CF evaluation may lead
* to singularities, I have replaced it with the modified Lentz algorithm
* given in
*
* I J Thompson and A R Barnett
* Coulomb and Bessel Functions of Complex Arguments and Order
* J Computational Physics 64:490-509 (1986)
*
* In consequence, gamma_inc_Q_CF_protected() is now obsolete and has been
* removed.
*
* Identification of terms between the above equation for F(a, x) and
* the first equation in the appendix of Thompson&Barnett is as follows:
*
*    b_0 = 0, b_n = 1 for all n > 0
*
*    a_1 = 1
*    a_n = (n/2-a)/x    for n even
*    a_n = (n-1)/(2x)   for n odd
*
*/

/* Continued fraction which occurs in evaluation
* of Q(a,x) or Gamma(a,x).
*
*              1   (1-a)/x  1/x  (2-a)/x   2/x  (3-a)/x
*   F(a,x) =  ---- ------- ----- -------- ----- -------- 
*             1 +   1 +     1 +   1 +      1 +   1 +
*
* Hans E. Plesser, 2002-01-22 (hans dot plesser at itf dot nlh dot no).
*
* Split out from gamma_inc_Q_CF() by GJ [Tue Apr  1 13:16:41 MST 2003].
* See gamma_inc_Q_CF() below.
*
*/
static int
gamma_inc_F_CF(double a, double x, gsl_sf_result * result)
{
	int    nmax = 5000;
	double small = gsl_pow_3(GSL_DBL_EPSILON);

	double hn = 1.0;           /* convergent */
	double Cn = 1.0 / small;
	double Dn = 1.0;
	int n;

	/* n == 1 has a_1, b_1, b_0 independent of a,x,
	so that has been done by hand                */
	for (n = 2; n < nmax; n++)
	{
		double an;
		double delta;

		if (GSL_IS_ODD(n))
			an = 0.5*(n - 1) / x;
		else
			an = (0.5*n - a) / x;

		Dn = 1.0 + an * Dn;
		if (fabs(Dn) < small)
			Dn = small;
		Cn = 1.0 + an / Cn;
		if (fabs(Cn) < small)
			Cn = small;
		Dn = 1.0 / Dn;
		delta = Cn * Dn;
		hn *= delta;
		if (fabs(delta - 1.0) < GSL_DBL_EPSILON) break;
	}

	result->val = hn;
	result->err = 2.0*GSL_DBL_EPSILON * fabs(hn);
	result->err += GSL_DBL_EPSILON * (2.0 + 0.5*n) * fabs(result->val);

	if (n == nmax)
		GSL_ERROR(GSL_EMAXITER);
	else
		return GSL_SUCCESS;
}

static
int
gamma_inc_Q_CF(double a, double x, gsl_sf_result * result)
{
	gsl_sf_result D;
	gsl_sf_result F;
	int stat_D = gamma_inc_D(a, x, &D);
	int stat_F = gamma_inc_F_CF(a, x, &F);

	result->val = D.val * (a / x) * F.val;
	result->err = D.err * fabs((a / x) * F.val) + fabs(D.val * a / x * F.err);

	return GSL_ERROR_SELECT_2(stat_F, stat_D);
}

/* Q large x asymptotic
*/
static
int
gamma_inc_Q_large_x(double a, double x, gsl_sf_result * result)
{
	int nmax = 5000;

	gsl_sf_result D;
	int stat_D = gamma_inc_D(a, x, &D);

	double sum = 1.0;
	double term = 1.0;
	double last = 1.0;
	int n;
	for (n = 1; n<nmax; n++) {
		term *= (a - n) / x;
		if (fabs(term / last) > 1.0) break;
		if (fabs(term / sum)  < GSL_DBL_EPSILON) break;
		sum += term;
		last = term;
	}

	result->val = D.val * (a / x) * sum;
	result->err = D.err * fabs((a / x) * sum);
	result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);

	if (n == nmax)
		GSL_ERROR(GSL_EMAXITER);
	else
		return stat_D;
}

int
gsl_sf_gamma_inc_Q_e(double a, double x, gsl_sf_result * result)
{
	if (a < 0.0 || x < 0.0) {
		DOMAIN_ERROR(result);
	}
	else if (x == 0.0) {
		result->val = 1.0;
		result->err = 0.0;
		return GSL_SUCCESS;
	}
	else if (a == 0.0)
	{
		result->val = 0.0;
		result->err = 0.0;
		return GSL_SUCCESS;
	}
	else if (x <= 0.5*a) {
		/* If the series is quick, do that. It is
		* robust and simple.
		*/
		gsl_sf_result P;
		int stat_P = gamma_inc_P_series(a, x, &P);
		result->val = 1.0 - P.val;
		result->err = P.err;
		result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
		return stat_P;
	}
	else if (a >= 1.0e+06 && (x - a)*(x - a) < a) {
		/* Then try the difficult asymptotic regime.
		* This is the only way to do this region.
		*/
		return gamma_inc_Q_asymp_unif(a, x, result);
	}
	else if (a < 0.2 && x < 5.0) {
		/* Cancellations at small a must be handled
		* analytically; x should not be too big
		* either since the series terms grow
		* with x and log(x).
		*/
		return gamma_inc_Q_series(a, x, result);
	}
	else if (a <= x) {
		if (x <= 1.0e+06) {
			/* Continued fraction is excellent for x >~ a.
			* We do not let x be too large when x > a since
			* it is somewhat pointless to try this there;
			* the function is rapidly decreasing for
			* x large and x > a, and it will just
			* underflow in that region anyway. We
			* catch that case in the standard
			* large-x method.
			*/
			return gamma_inc_Q_CF(a, x, result);
		}
		else {
			return gamma_inc_Q_large_x(a, x, result);
		}
	}
	else {
		if (x > a - sqrt(a)) {
			/* Continued fraction again. The convergence
			* is a little slower here, but that is fine.
			* We have to trade that off against the slow
			* convergence of the series, which is the
			* only other option.
			*/
			return gamma_inc_Q_CF(a, x, result);
		}
		else {
			gsl_sf_result P;
			int stat_P = gamma_inc_P_series(a, x, &P);
			result->val = 1.0 - P.val;
			result->err = P.err;
			result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
			return stat_P;
		}
	}
}

static double
beta_cont_frac(double a, double b, double x,
	double epsabs)
{
	unsigned int max_iter = 512;    /* control iterations      */
	double cutoff = 2.0 * GSL_DBL_MIN;      /* control the zero cutoff */
	unsigned int iter_count = 0;
	double cf;

	/* standard initialization for continued fraction */
	double num_term = 1.0;
	double den_term = 1.0 - (a + b) * x / (a + 1.0);

	if (fabs(den_term) < cutoff)
		den_term = GSL_NAN;

	den_term = 1.0 / den_term;
	cf = den_term;

	while (iter_count < max_iter)
	{
		int k = iter_count + 1;
		double coeff = k * (b - k) * x / (((a - 1.0) + 2 * k) * (a + 2 * k));
		double delta_frac;

		/* first step */
		den_term = 1.0 + coeff * den_term;
		num_term = 1.0 + coeff / num_term;

		if (fabs(den_term) < cutoff)
			den_term = GSL_NAN;

		if (fabs(num_term) < cutoff)
			num_term = GSL_NAN;

		den_term = 1.0 / den_term;

		delta_frac = den_term * num_term;
		cf *= delta_frac;

		coeff = -(a + k) * (a + b + k) * x / ((a + 2 * k) * (a + 2 * k + 1.0));

		/* second step */
		den_term = 1.0 + coeff * den_term;
		num_term = 1.0 + coeff / num_term;

		if (fabs(den_term) < cutoff)
			den_term = GSL_NAN;

		if (fabs(num_term) < cutoff)
			num_term = GSL_NAN;

		den_term = 1.0 / den_term;

		delta_frac = den_term * num_term;
		cf *= delta_frac;

		if (fabs(delta_frac - 1.0) < 2.0 * GSL_DBL_EPSILON)
			break;

		if (cf * fabs(delta_frac - 1.0) < epsabs)
			break;

		++iter_count;
	}

	if (iter_count >= max_iter)
		return GSL_NAN;

	return cf;
}

double gsl_sf_gamma_inc_Q(double a, double x)
{
	EVAL_RESULT(gsl_sf_gamma_inc_Q_e(a, x, &result));
}

int
gsl_sf_gamma_inc_P_e(double a, double x, gsl_sf_result * result)
{
	if (a <= 0.0 || x < 0.0) {
		DOMAIN_ERROR(result);
	}
	else if (x == 0.0) {
		result->val = 0.0;
		result->err = 0.0;
		return GSL_SUCCESS;
	}
	else if (x < 20.0 || x < 0.5*a) {
		/* Do the easy series cases. Robust and quick.
		*/
		return gamma_inc_P_series(a, x, result);
	}
	else if (a > 1.0e+06 && (x - a)*(x - a) < a) {
		/* Crossover region. Note that Q and P are
		* roughly the same order of magnitude here,
		* so the subtraction is stable.
		*/
		gsl_sf_result Q;
		int stat_Q = gamma_inc_Q_asymp_unif(a, x, &Q);
		result->val = 1.0 - Q.val;
		result->err = Q.err;
		result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
		return stat_Q;
	}
	else if (a <= x) {
		/* Q <~ P in this area, so the
		* subtractions are stable.
		*/
		gsl_sf_result Q;
		int stat_Q;
		if (a > 0.2*x) {
			stat_Q = gamma_inc_Q_CF(a, x, &Q);
		}
		else {
			stat_Q = gamma_inc_Q_large_x(a, x, &Q);
		}
		result->val = 1.0 - Q.val;
		result->err = Q.err;
		result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
		return stat_Q;
	}
	else {
		if ((x - a)*(x - a) < a) {
			/* This condition is meant to insure
			* that Q is not very close to 1,
			* so the subtraction is stable.
			*/
			gsl_sf_result Q;
			int stat_Q = gamma_inc_Q_CF(a, x, &Q);
			result->val = 1.0 - Q.val;
			result->err = Q.err;
			result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
			return stat_Q;
		}
		else {
			return gamma_inc_P_series(a, x, result);
		}
	}
}



double gsl_sf_gamma_inc_P(double a, double x)
{
	EVAL_RESULT(gsl_sf_gamma_inc_P_e(a, x, &result));
}

static double
isnegint(double x)
{
	return (x < 0) && (x == floor(x));
}

/*-*-*-*-*-*-*-*-*-*-*-* Private Section *-*-*-*-*-*-*-*-*-*-*-*/

/* Chebyshev expansion for log(1 + x(t))/x(t)
*
* x(t) = (4t-1)/(2(4-t))
* t(x) = (8x+1)/(2(x+2))
* -1/2 < x < 1/2
* -1 < t < 1
*/
__constant double lopx_data[21] = {
	2.16647910664395270521272590407,
	-0.28565398551049742084877469679,
	0.01517767255690553732382488171,
	-0.00200215904941415466274422081,
	0.00019211375164056698287947962,
	-0.00002553258886105542567601400,
	2.9004512660400621301999384544e-06,
	-3.8873813517057343800270917900e-07,
	4.7743678729400456026672697926e-08,
	-6.4501969776090319441714445454e-09,
	8.2751976628812389601561347296e-10,
	-1.1260499376492049411710290413e-10,
	1.4844576692270934446023686322e-11,
	-2.0328515972462118942821556033e-12,
	2.7291231220549214896095654769e-13,
	-3.7581977830387938294437434651e-14,
	5.1107345870861673561462339876e-15,
	-7.0722150011433276578323272272e-16,
	9.7089758328248469219003866867e-17,
	-1.3492637457521938883731579510e-17,
	1.8657327910677296608121390705e-18
};
__constant cheb_series lopx_cs = {
	lopx_data,
	20,
	-1, 1,
	10
};

int
gsl_sf_log_1plusx_e(double x, gsl_sf_result * result)
{
	/* CHECK_POINTER(result) */

	if (x <= -1.0) {
		DOMAIN_ERROR(result);
	}
	else if (fabs(x) < GSL_ROOT6_DBL_EPSILON) {
		double c1 = -0.5;
		double c2 = 1.0 / 3.0;
		double c3 = -1.0 / 4.0;
		double c4 = 1.0 / 5.0;
		double c5 = -1.0 / 6.0;
		double c6 = 1.0 / 7.0;
		double c7 = -1.0 / 8.0;
		double c8 = 1.0 / 9.0;
		double c9 = -1.0 / 10.0;
		double t = c5 + x*(c6 + x*(c7 + x*(c8 + x*c9)));
		result->val = x * (1.0 + x*(c1 + x*(c2 + x*(c3 + x*(c4 + x*t)))));
		result->err = GSL_DBL_EPSILON * fabs(result->val);
		return GSL_SUCCESS;
	}
	else if (fabs(x) < 0.5) {
		double t = 0.5*(8.0*x + 1.0) / (x + 2.0);
		gsl_sf_result c;
		cheb_eval_e(&lopx_cs, t, &c);
		result->val = x * c.val;
		result->err = fabs(x * c.err);
		return GSL_SUCCESS;
	}
	else {
		result->val = log(1.0 + x);
		result->err = GSL_DBL_EPSILON * fabs(result->val);
		return GSL_SUCCESS;
	}
}


int
gsl_sf_lnbeta_sgn_e(double x, double y, gsl_sf_result * result, double * sgn)
{
	/* CHECK_POINTER(result) */

	if (x == 0.0 || y == 0.0) {
		*sgn = 0.0;
		DOMAIN_ERROR(result);
	}
	else if (isnegint(x) || isnegint(y)) {
		*sgn = 0.0;
		DOMAIN_ERROR(result); /* not defined for negative integers */
	}

	/* See if we can handle the postive case with min/max < 0.2 */

	if (x > 0 && y > 0) {
		double max = GSL_MAX(x, y);
		double min = GSL_MIN(x, y);
		double rat = min / max;

		if (rat < 0.2) {
			/* min << max, so be careful
			* with the subtraction
			*/
			double lnpre_val;
			double lnpre_err;
			double lnpow_val;
			double lnpow_err;
			double t1, t2, t3;
			gsl_sf_result lnopr;
			gsl_sf_result gsx, gsy, gsxy;
			gsl_sf_gammastar_e(x, &gsx);
			gsl_sf_gammastar_e(y, &gsy);
			gsl_sf_gammastar_e(x + y, &gsxy);
			gsl_sf_log_1plusx_e(rat, &lnopr);
			lnpre_val = log(gsx.val*gsy.val / gsxy.val * M_SQRT2*M_SQRTPI);
			lnpre_err = gsx.err / gsx.val + gsy.err / gsy.val + gsxy.err / gsxy.val;
			t1 = min*log(rat);
			t2 = 0.5*log(min);
			t3 = (x + y - 0.5)*lnopr.val;
			lnpow_val = t1 - t2 - t3;
			lnpow_err = GSL_DBL_EPSILON * (fabs(t1) + fabs(t2) + fabs(t3));
			lnpow_err += fabs(x + y - 0.5) * lnopr.err;
			result->val = lnpre_val + lnpow_val;
			result->err = lnpre_err + lnpow_err;
			result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
			*sgn = 1.0;
			return GSL_SUCCESS;
		}
	}

	/* General case - Fallback */
	{
		gsl_sf_result lgx, lgy, lgxy;
		double sgx, sgy, sgxy, xy = x + y;
		int stat_gx = gsl_sf_lngamma_sgn_e(x, &lgx, &sgx);
		int stat_gy = gsl_sf_lngamma_sgn_e(y, &lgy, &sgy);
		int stat_gxy = gsl_sf_lngamma_sgn_e(xy, &lgxy, &sgxy);
		*sgn = sgx * sgy * sgxy;
		result->val = lgx.val + lgy.val - lgxy.val;
		result->err = lgx.err + lgy.err + lgxy.err;
		result->err += 2.0 * GSL_DBL_EPSILON * (fabs(lgx.val) + fabs(lgy.val) + fabs(lgxy.val));
		result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
		return GSL_ERROR_SELECT_3(stat_gx, stat_gy, stat_gxy);
	}
}

int
gsl_sf_lnbeta_e(double x, double y, gsl_sf_result * result)
{
	double sgn;
	int status = gsl_sf_lnbeta_sgn_e(x, y, result, &sgn);
	if (sgn == -1) {
		DOMAIN_ERROR(result);
	}
	return status;
}

int gsl_sf_lngamma_sgn_e(double x, gsl_sf_result * result_lg, double * sgn)
{
	if (fabs(x - 1.0) < 0.01) {
		int stat = lngamma_1_pade(x - 1.0, result_lg);
		result_lg->err *= 1.0 / (GSL_DBL_EPSILON + fabs(x - 1.0));
		*sgn = 1.0;
		return stat;
	}
	else if (fabs(x - 2.0) < 0.01) {
		int stat = lngamma_2_pade(x - 2.0, result_lg);
		result_lg->err *= 1.0 / (GSL_DBL_EPSILON + fabs(x - 2.0));
		*sgn = 1.0;
		return stat;
	}
	else if (x >= 0.5) {
		*sgn = 1.0;
		return lngamma_lanczos(x, result_lg);
	}
	else if (x == 0.0) {
		*sgn = 0.0;
		DOMAIN_ERROR(result_lg);
	}
	else if (fabs(x) < 0.02) {
		return lngamma_sgn_0(x, result_lg, sgn);
	}
	else if (x > -0.5 / (GSL_DBL_EPSILON*M_PI)) {
		/* Try to extract a fractional
		* part from x.
		*/
		double z = 1.0 - x;
		double s = sin(M_PI*x);
		double as = fabs(s);
		if (s == 0.0) {
			*sgn = 0.0;
			DOMAIN_ERROR(result_lg);
		}
		else if (as < M_PI*0.015) {
			/* x is near a negative integer, -N */
			if (x < INT_MIN + 2.0) {
				result_lg->val = 0.0;
				result_lg->err = 0.0;
				*sgn = 0.0;
				GSL_ERROR(GSL_EROUND);
			}
			else {
				int N = -(int)(x - 0.5);
				double eps = x + N;
				return lngamma_sgn_sing(N, eps, result_lg, sgn);
			}
		}
		else {
			gsl_sf_result lg_z;
			lngamma_lanczos(z, &lg_z);
			*sgn = (s > 0.0 ? 1.0 : -1.0);
			result_lg->val = M_LNPI - (log(as) + lg_z.val);
			result_lg->err = 2.0 * GSL_DBL_EPSILON * fabs(result_lg->val) + lg_z.err;
			return GSL_SUCCESS;
		}
	}
	else {
		/* |x| was too large to extract any fractional part */
		result_lg->val = 0.0;
		result_lg->err = 0.0;
		*sgn = 0.0;
		GSL_ERROR(GSL_EROUND);
	}
}

double gsl_sf_lnbeta(double x, double y)
{
	EVAL_RESULT(gsl_sf_lnbeta_e(x, y, &result));
}

static double
beta_inc_AXPY(double A, double Y,
	double a, double b, double x)
{
	if (x == 0.0)
	{
		return A * 0 + Y;
	}
	else if (x == 1.0)
	{
		return A * 1 + Y;
	}
	else if (a > 1e5 && b < 10 && x > a / (a + b))
	{
		/* Handle asymptotic regime, large a, small b, x > peak [AS 26.5.17] */
		double N = a + (b - 1.0) / 2.0;
		return A * gsl_sf_gamma_inc_Q(b, -N * log(x)) + Y;
	}
	else if (b > 1e5 && a < 10 && x < b / (a + b))
	{
		/* Handle asymptotic regime, small a, large b, x < peak [AS 26.5.17] */
		double N = b + (a - 1.0) / 2.0;
		return A * gsl_sf_gamma_inc_P(a, -N * log1p(-x)) + Y;
	}
	else
	{
		double ln_beta = gsl_sf_lnbeta(a, b);
		double ln_pre = -ln_beta + a * log(x) + b * log1p(-x);

		double prefactor = exp(ln_pre);

		if (x < (a + 1.0) / (a + b + 2.0))
		{
			/* Apply continued fraction directly. */
			double epsabs = fabs(Y / (A * prefactor / a)) * GSL_DBL_EPSILON;

			double cf = beta_cont_frac(a, b, x, epsabs);

			return A * (prefactor * cf / a) + Y;
		}
		else
		{
			/* Apply continued fraction after hypergeometric transformation. */
			double epsabs =
				fabs((A + Y) / (A * prefactor / b)) * GSL_DBL_EPSILON;
			double cf = beta_cont_frac(b, a, 1.0 - x, epsabs);
			double term = prefactor * cf / b;

			if (A == -Y)
			{
				return -A * term;
			}
			else
			{
				return A * (1 - term) + Y;
			}
		}
	}
}

double
gsl_cdf_beta_P(double x, double a, double b)
{
	double P;

	if (x <= 0.0)
	{
		return 0.0;
	}

	if (x >= 1.0)
	{
		return 1.0;
	}

	P = beta_inc_AXPY(1.0, 0.0, a, b, x);

	return P;
}

double
gsl_cdf_beta_Q(double x, double a, double b)
{
	double Q;

	if (x >= 1.0)
	{
		return 0.0;
	}

	if (x <= 0.0)
	{
		return 1.0;
	}

	Q = beta_inc_AXPY(-1.0, 1.0, a, b, x);

	return Q;
}

struct gsl_cdf_beta_request
{
	double x, a, b;
};

typedef struct gsl_cdf_beta_request gsl_cdf_beta_request;

struct gsl_cdf_beta_response
{
	double result;
};

typedef struct gsl_cdf_beta_response gsl_cdf_beta_response;

__kernel void gsl_cdf_beta_P_cl(__global gsl_cdf_beta_request *request, __global gsl_cdf_beta_response *response)
{
	int threadId = get_global_id(0);

	response[threadId].result = gsl_cdf_beta_P(request[threadId].x, request[threadId].a, request[threadId].b);
}

__kernel void gsl_cdf_beta_Q_cl(__global gsl_cdf_beta_request *request, __global gsl_cdf_beta_response *response)
{
	int threadId = get_global_id(0);

	response[threadId].result = gsl_cdf_beta_Q(request[threadId].x, request[threadId].a, request[threadId].b);
}
