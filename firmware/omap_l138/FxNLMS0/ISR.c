/*
 * ISR.c
 *
 * Real-Time Active Noise Control (ANC) with Filtered-X Normalized LMS (FxNLMS)
 * Target: TI OMAP-L138 / TMS320C6748
 *
 * Inputs:
 *   - Line-in Left  : Reference Microphone x(n)
 *   - Line-in Right : Error Microphone e(n)
 * Output:
 *   - Line-out Right: Canceling Anti-Noise (-y(n)) [Left muted for TPA3116D2 protection]
 */

#include <xdc/std.h>
#include <xdc/runtime/System.h>
#include <ti/sysbios/BIOS.h>
#include <ti/sysbios/knl/Swi.h>
#include <xdc/cfg/global.h>
#include <math.h>
#include "LED_DIPSW.h"
#include "Timer.h"
#include "interrupt.h"
#include "McASP.h"
#include "define.h"
#include <c6x.h>

/* LED counter */
volatile static int32_t led_cnt = 0;

/* =========================================================================
 * FxNLMS ANC Buffers and Filter Definitions
 * ========================================================================= */

/* Primary adaptive filter W(z) - Length: FILTER_TAP (256) */
float W[FILTER_TAP] = {0.0f};

/* Reference microphone delay line: x(n), x(n-1), ... */
float X[FILTER_TAP] = {0.0f};

/*
 * Secondary path estimated model S_hat(z) coefficients (Length: SEC_FILTER_TAP = 500)
 * Measured from real hardware acoustic duct via NLMS0
 */
float S_hat[SEC_FILTER_TAP] = {
     0.00007884f,  0.00000642f,  0.00002148f, -0.00001800f,  0.00003429f,  0.00009421f,  0.00003917f,  0.00008757f,
     0.00005265f,  0.00008623f,  0.00008687f,  0.00016993f,  0.00010240f,  0.00006075f,  0.00011662f,  0.00003255f,
     0.00014089f,  0.00001210f,  0.00004105f,  0.00005980f, -0.00005111f,  0.00013254f,  0.00000331f,  0.00001902f,
    -0.00001913f, -0.00002770f,  0.00004881f, -0.00002999f,  0.00002192f, -0.00003756f, -0.00001484f,  0.00000105f,
     0.00002728f,  0.00000693f, -0.00002268f,  0.00001039f, -0.00001209f,  0.00009317f, -0.00000043f,  0.00001760f,
     0.00016772f, -0.00015568f,  0.00047579f, -0.00051359f,  0.00100319f, -0.00155505f,  0.00864968f,  0.01783636f,
     0.01180146f,  0.01850763f,  0.01271287f,  0.00639585f,  0.00897995f,  0.01057158f, -0.00602634f, -0.01166557f,
     0.00228828f, -0.00661336f, -0.01417176f, -0.00324677f, -0.00646436f, -0.01569409f, -0.00141600f, -0.01683539f,
    -0.00652382f, -0.01183275f, -0.01500575f, -0.00347425f, -0.01778132f,  0.00103329f, -0.01516917f, -0.00545452f,
     0.00316252f, -0.00639700f,  0.00382076f, -0.00099155f,  0.00046380f, -0.00279948f, -0.00051500f, -0.00502292f,
    -0.00269887f,  0.00774390f, -0.00082042f,  0.01093068f,  0.00119602f,  0.00714211f,  0.00308746f,  0.00092500f,
     0.01479592f, -0.00676373f,  0.00787278f,  0.00091632f,  0.00192427f,  0.00744333f,  0.00013506f,  0.00940553f,
    -0.00021571f,  0.00602590f,  0.00187296f,  0.00318058f,  0.00418777f,  0.00060464f,  0.00541214f, -0.00197467f,
     0.00491940f, -0.00134745f,  0.00362121f,  0.00076989f, -0.00237688f,  0.00563353f, -0.00428163f,  0.00246886f,
    -0.00473780f, -0.00102513f, -0.00049974f, -0.00375120f,  0.00366714f, -0.00258771f,  0.00013733f, -0.00236501f,
    -0.00020779f, -0.00141886f, -0.00323906f,  0.00045784f, -0.00344508f, -0.00051454f, -0.00310000f, -0.00021800f,
    -0.00003448f, -0.00210630f,  0.00170022f, -0.00294486f,  0.00056650f, -0.00265423f,  0.00029844f,  0.00115840f,
    -0.00201743f,  0.00236400f, -0.00084869f,  0.00144587f, -0.00064612f,  0.00127749f,  0.00156107f,  0.00141156f,
     0.00416059f,  0.00440586f,  0.00837625f,  0.00481035f,  0.00525119f,  0.00083698f, -0.00223610f, -0.00240822f,
    -0.00518970f, -0.00137928f, -0.00395481f, -0.00353431f, -0.00281597f, -0.00463422f, -0.00255681f, -0.00294233f,
    -0.00020527f, -0.00104368f, -0.00615958f, -0.00467214f, -0.00638514f, -0.00643539f, -0.00604312f, -0.00375140f,
    -0.00463437f, -0.00631395f, -0.00027716f,  0.00251047f,  0.00607609f,  0.00227223f,  0.00197376f,  0.00234679f,
     0.00051947f,  0.00575221f,  0.00596073f,  0.00990534f,  0.00770235f,  0.01027798f,  0.00901424f,  0.00618521f,
     0.01047776f,  0.00370516f,  0.00443000f,  0.00479756f,  0.00402900f,  0.00412016f,  0.00093414f,  0.00131095f,
     0.00188946f,  0.00344188f,  0.00074859f, -0.00001659f, -0.00107206f, -0.00422317f, -0.00327365f, -0.00379710f,
    -0.00217640f, -0.00386438f, -0.00171369f, -0.00332093f, -0.00572368f,  0.00098064f, -0.00235157f, -0.00130397f,
    -0.00197112f, -0.00197878f, -0.00303331f, -0.00470406f, -0.00002203f, -0.00291577f,  0.00191946f, -0.00095115f,
    -0.00130492f,  0.00023280f, -0.00042534f, -0.00092758f, -0.00277247f, -0.00035519f, -0.00146925f, -0.00082574f,
    -0.00007863f, -0.00008544f,  0.00121096f,  0.00104219f,  0.00099307f, -0.00075535f,  0.00029214f,  0.00020930f,
    -0.00162068f, -0.00108880f,  0.00097140f,  0.00224842f, -0.00073191f, -0.00103902f,  0.00027795f, -0.00075547f,
     0.00050112f, -0.00081320f,  0.00063208f, -0.00087815f, -0.00216324f,  0.00024858f, -0.00066923f,  0.00021774f,
     0.00078473f,  0.00017633f, -0.00006323f, -0.00057881f,  0.00054949f, -0.00043166f, -0.00082234f,  0.00134757f,
    -0.00097067f,  0.00245324f, -0.00044688f,  0.00226823f,  0.00047453f,  0.00030769f,  0.00296918f, -0.00155734f,
     0.00244092f, -0.00183249f,  0.00099398f,  0.00115308f,  0.00021739f,  0.00045679f,  0.00096942f,  0.00122751f,
    -0.00108707f,  0.00123470f, -0.00003773f,  0.00130701f,  0.00028034f,  0.00145196f,  0.00077808f,  0.00026367f,
     0.00136254f, -0.00004598f,  0.00002017f,  0.00071539f,  0.00115488f,  0.00139344f, -0.00051892f,  0.00058770f,
    -0.00004151f, -0.00201727f,  0.00042053f, -0.00132459f, -0.00067054f, -0.00226254f, -0.00261257f, -0.00238441f,
    -0.00273505f, -0.00278425f, -0.00189085f, -0.00228132f, -0.00208675f, -0.00209277f, -0.00176614f, -0.00110710f,
    -0.00191087f, -0.00024364f, -0.00164591f, -0.00147989f, -0.00144930f,  0.00019879f,  0.00089676f,  0.00030257f,
     0.00173125f,  0.00094566f,  0.00138600f,  0.00178470f,  0.00312540f,  0.00055630f,  0.00106096f,  0.00223092f,
     0.00205161f,  0.00230551f,  0.00053491f,  0.00185878f,  0.00058155f,  0.00118322f,  0.00181936f,  0.00026081f,
    -0.00024996f, -0.00056177f,  0.00101202f,  0.00152520f, -0.00005414f,  0.00017410f, -0.00029246f,  0.00034854f,
     0.00068803f, -0.00085049f,  0.00100196f,  0.00021975f,  0.00083374f,  0.00021643f, -0.00104115f,  0.00009095f,
    -0.00004751f,  0.00085221f, -0.00009941f,  0.00089693f, -0.00081479f,  0.00080475f, -0.00011715f,  0.00087971f,
    -0.00032827f, -0.00007978f,  0.00083900f, -0.00050570f,  0.00165170f, -0.00130158f,  0.00010746f,  0.00001617f,
     0.00027573f,  0.00127543f, -0.00067690f,  0.00082006f, -0.00028613f,  0.00041499f,  0.00063185f, -0.00003752f,
     0.00038031f, -0.00076879f, -0.00020810f,  0.00004032f,  0.00078485f, -0.00115388f,  0.00039707f, -0.00103622f,
    -0.00052812f, -0.00059333f, -0.00098358f,  0.00028637f, -0.00211547f,  0.00049116f, -0.00111640f, -0.00086228f,
    -0.00016756f, -0.00101670f,  0.00013601f, -0.00084782f,  0.00026186f, -0.00038395f, -0.00063531f, -0.00028426f,
    -0.00018562f,  0.00046654f, -0.00054331f,  0.00018256f,  0.00018393f,  0.00037033f, -0.00052021f,  0.00018314f,
    -0.00014238f, -0.00025889f,  0.00034069f, -0.00046969f,  0.00016056f, -0.00036271f,  0.00046265f,  0.00006380f,
     0.00076337f,  0.00048880f,  0.00033405f,  0.00011587f,  0.00009759f,  0.00067847f,  0.00068119f,  0.00065927f,
     0.00033469f,  0.00133744f,  0.00065512f,  0.00039765f,  0.00071094f,  0.00011384f,  0.00063447f, -0.00000634f,
     0.00099604f, -0.00013330f,  0.00028579f,  0.00016255f, -0.00031021f,  0.00079378f, -0.00076880f,  0.00023667f,
    -0.00127774f, -0.00045186f, -0.00060563f, -0.00093056f, -0.00055180f, -0.00135560f, -0.00028201f, -0.00117362f,
     0.00012161f, -0.00074216f, -0.00032250f, -0.00036087f, -0.00078979f,  0.00075343f, -0.00133581f, -0.00000285f,
    -0.00062251f,  0.00022544f,  0.00035493f, -0.00043415f,  0.00090761f, -0.00053294f,  0.00073138f, -0.00008275f,
     0.00010616f,  0.00020683f,  0.00041940f,  0.00061877f, -0.00019226f,  0.00067631f, -0.00037364f,  0.00112736f,
    -0.00055511f,  0.00010210f,  0.00032719f, -0.00076168f,  0.00104071f, -0.00054591f,  0.00110205f, -0.00033692f,
    -0.00016384f,  0.00072431f, -0.00065310f,  0.00061961f, -0.00039739f,  0.00037032f, -0.00006108f,  0.00010962f,
     0.00026448f, -0.00009076f,  0.00009896f, -0.00007171f,  0.00063773f, -0.00031415f,  0.00071480f, -0.00004586f,
     0.00006783f,  0.00060958f, -0.00011436f,  0.00088361f,  0.00021008f,  0.00060757f,  0.00083875f,  0.00018623f,
     0.00054020f,  0.00024005f,  0.00051208f,  0.00051361f
};

/*
 * Acoustic Feedback Path estimated model F_hat(z) coefficients (Length: FB_FILTER_TAP = 500)
 * Measured from real hardware acoustic duct via NLMS0 (Canceling Speaker -> Reference Mic)
 */
float F_hat[FB_FILTER_TAP] = {
     0.00003626f,  0.00003986f,  0.00005326f,  0.00004775f,  0.00004335f,  0.00003797f,  0.00006325f,  0.00000973f, 
     0.00007527f,  0.00005732f,  0.00004307f,  0.00007973f,  0.00003505f,  0.00008803f,  0.00004335f,  0.00005028f, 
     0.00007125f,  0.00005040f,  0.00006333f,  0.00007168f,  0.00008282f,  0.00005615f,  0.00009075f,  0.00005369f, 
     0.00006159f,  0.00007402f,  0.00007520f,  0.00008578f,  0.00005476f,  0.00008308f,  0.00008099f,  0.00006175f, 
     0.00013751f,  0.00006336f,  0.00012208f,  0.00005010f,  0.00010724f,  0.00009001f,  0.00005336f,  0.00012832f, 
     0.00007678f,  0.00011567f,  0.00006744f,  0.00009875f,  0.00009733f,  0.00004532f,  0.00008876f,  0.00005142f, 
     0.00008013f,  0.00003303f,  0.00006984f,  0.00007656f,  0.00000678f,  0.00006468f,  0.00003488f,  0.00008826f, 
     0.00001599f,  0.00008399f,  0.00008110f,  0.00004608f,  0.00010337f,  0.00006185f,  0.00008513f,  0.00006715f, 
     0.00008831f,  0.00010318f,  0.00006050f,  0.00009823f,  0.00009613f,  0.00010946f,  0.00005326f,  0.00009483f, 
     0.00015407f,  0.00008582f,  0.00007561f,  0.00009049f,  0.00007513f,  0.00012410f,  0.00013426f,  0.00007614f, 
     0.00004780f,  0.00008332f,  0.00006412f,  0.00014485f,  0.00003534f,  0.00035814f,  0.00229546f,  0.00545922f, 
     0.00687100f,  0.00696966f,  0.00589946f,  0.00423670f,  0.00226803f,  0.00203304f,  0.00137635f,  0.00215026f, 
     0.00298142f,  0.00241872f,  0.00323051f,  0.00368553f,  0.00510284f, -0.00000735f, -0.00244934f, -0.00516430f, 
    -0.00567674f, -0.00591219f, -0.00706539f, -0.00775945f, -0.00999246f, -0.01127892f, -0.00770105f, -0.00516952f, 
    -0.00624144f, -0.00273455f, -0.00959673f, -0.00929576f, -0.00977820f, -0.00965766f, -0.00408048f, -0.00506221f, 
    -0.00443246f, -0.00370112f, -0.00070791f,  0.00366614f,  0.00347057f,  0.00206354f,  0.00135257f,  0.00447335f, 
     0.00252578f,  0.00253940f,  0.00454994f,  0.00459240f,  0.00670271f,  0.00325901f,  0.01100282f,  0.00659425f, 
     0.00635281f,  0.00595900f,  0.00461968f,  0.00295167f,  0.00282615f,  0.00653116f,  0.00053907f,  0.00436349f, 
     0.00394052f,  0.00107421f,  0.00286630f,  0.00467944f,  0.00277316f, -0.00206142f,  0.00145224f,  0.00221869f, 
    -0.00333115f,  0.00134465f,  0.00057435f, -0.00127170f,  0.00003222f, -0.00018682f, -0.00044587f, -0.00167184f, 
     0.00118921f, -0.00255901f, -0.00286403f, -0.00207748f, -0.00340481f,  0.00013811f, -0.00268529f,  0.00036477f, 
    -0.00085543f, -0.00202004f,  0.00101594f, -0.00274526f, -0.00028258f, -0.00202877f, -0.00369922f,  0.00119741f, 
     0.00021454f, -0.00132085f,  0.00092754f,  0.00120165f, -0.00010604f,  0.00011352f,  0.00069090f, -0.00036452f, 
    -0.00379427f,  0.00191463f,  0.00014256f, -0.00062256f,  0.00165395f,  0.00094394f,  0.00253202f, -0.00173741f, 
     0.00250768f, -0.00101957f, -0.00134983f,  0.00081445f, -0.00070931f,  0.00031002f,  0.00035455f,  0.00148310f, 
     0.00053178f,  0.00041339f, -0.00126176f,  0.00105029f, -0.00176632f, -0.00116021f,  0.00098819f, -0.00115417f, 
     0.00138073f, -0.00041888f,  0.00165828f,  0.00012040f, -0.00028413f,  0.00108969f, -0.00029011f, -0.00057743f, 
     0.00010366f,  0.00115080f,  0.00165863f,  0.00067690f,  0.00318041f,  0.00253145f,  0.00082506f,  0.00322763f, 
     0.00240155f,  0.00104842f,  0.00229580f,  0.00045583f,  0.00259978f,  0.00081011f,  0.00158995f,  0.00323764f, 
     0.00037815f,  0.00262860f,  0.00133444f,  0.00068719f,  0.00035499f, -0.00114534f, -0.00022265f, -0.00254260f, 
    -0.00183802f, -0.00105104f, -0.00073057f, -0.00099322f, -0.00164156f, -0.00221705f, -0.00249549f, -0.00232579f, 
    -0.00193471f, -0.00058918f, -0.00289218f, -0.00248206f, -0.00242138f, -0.00294101f, -0.00088936f, -0.00180692f, 
    -0.00017637f, -0.00094462f, -0.00096533f, -0.00026652f, -0.00072234f, -0.00019317f, -0.00149688f, -0.00051766f, 
    -0.00046021f,  0.00072095f,  0.00016433f,  0.00011139f,  0.00041411f, -0.00023471f,  0.00154523f, -0.00034912f, 
     0.00085673f,  0.00047022f, -0.00041691f,  0.00053631f, -0.00101798f,  0.00151526f,  0.00085150f,  0.00023113f, 
     0.00094813f,  0.00013661f,  0.00035765f,  0.00003933f,  0.00103681f,  0.00014405f,  0.00015470f, -0.00041944f, 
     0.00034505f,  0.00149906f,  0.00000826f,  0.00109274f, -0.00074339f,  0.00066491f,  0.00019543f,  0.00010348f, 
     0.00100916f, -0.00073661f,  0.00098448f,  0.00012296f,  0.00074941f,  0.00140554f, -0.00013248f,  0.00049865f, 
    -0.00023809f,  0.00114993f,  0.00065041f,  0.00049455f,  0.00032706f,  0.00044878f,  0.00113874f,  0.00009739f, 
     0.00146618f, -0.00017675f,  0.00096432f,  0.00011527f, -0.00023730f,  0.00106586f, -0.00059272f,  0.00060603f, 
    -0.00045763f,  0.00019781f,  0.00042519f, -0.00080355f,  0.00066727f, -0.00077274f,  0.00023265f, -0.00042549f, 
    -0.00058098f, -0.00005854f, -0.00073488f, -0.00014328f, -0.00063776f,  0.00068578f, -0.00106769f, -0.00037863f, 
     0.00000691f, -0.00076838f,  0.00042525f, -0.00145116f,  0.00035717f, -0.00019962f, -0.00006175f,  0.00067873f, 
    -0.00089318f,  0.00037539f, -0.00035997f, -0.00006569f,  0.00021168f, -0.00028577f,  0.00030848f,  0.00010915f, 
     0.00064930f,  0.00028824f,  0.00038127f,  0.00028157f,  0.00099005f,  0.00043106f,  0.00043169f,  0.00131961f, 
    -0.00062749f,  0.00136941f,  0.00034917f,  0.00075185f,  0.00111359f,  0.00001623f,  0.00155019f,  0.00028934f, 
     0.00114686f,  0.00078606f,  0.00010643f,  0.00075743f,  0.00013586f,  0.00069657f,  0.00032193f,  0.00040331f, 
    -0.00051708f,  0.00050637f,  0.00024055f, -0.00014965f,  0.00114960f, -0.00052975f,  0.00014252f, -0.00046905f, 
    -0.00049924f,  0.00081189f, -0.00094739f,  0.00042961f, -0.00059976f, -0.00047564f,  0.00027663f, -0.00041413f, 
     0.00050774f, -0.00071495f,  0.00030400f, -0.00041246f, -0.00010488f, -0.00040520f, -0.00058995f,  0.00007449f, 
    -0.00029710f,  0.00046869f, -0.00064356f,  0.00059895f, -0.00016781f, -0.00080385f,  0.00056838f, -0.00115067f, 
     0.00044548f, -0.00013546f, -0.00022641f,  0.00026282f, -0.00099839f,  0.00016708f, -0.00021689f,  0.00003627f, 
    -0.00019112f, -0.00037598f, -0.00053125f, -0.00026570f, -0.00002367f, -0.00073460f,  0.00034001f, -0.00057058f, 
    -0.00017353f, -0.00006906f, -0.00046338f,  0.00059296f, -0.00103594f,  0.00018293f,  0.00028499f, -0.00052737f, 
     0.00097320f, -0.00032798f,  0.00065530f, -0.00015788f,  0.00017725f,  0.00089511f,  0.00015753f,  0.00050444f, 
     0.00021162f,  0.00064741f,  0.00010366f,  0.00099188f,  0.00057850f,  0.00002398f,  0.00049735f,  0.00015111f, 
     0.00116629f, -0.00020536f,  0.00014053f,  0.00091684f, -0.00046220f,  0.00076418f, -0.00032855f,  0.00030508f, 
     0.00017160f, -0.00042367f,  0.00067545f, -0.00026891f, -0.00015879f,  0.00029216f, -0.00002585f,  0.00000144f, 
    -0.00001003f,  0.00001134f, -0.00016944f,  0.00011520f, -0.00010345f,  0.00011487f,  0.00003309f, -0.00012753f, 
     0.00062566f, -0.00058373f,  0.00043186f, -0.00017168f, -0.00008212f,  0.00020696f, -0.00031361f,  0.00069486f, 
    -0.00031689f,  0.00021264f,  0.00013443f, -0.00018534f,  0.00007868f, -0.00012523f,  0.00000561f, -0.00013698f, 
     0.00016052f,  0.00009751f,  0.00011383f,  0.00007689f, -0.00015648f,  0.00040684f, -0.00050276f,  0.00026576f, 
    -0.00004236f, -0.00018262f,  0.00011321f, -0.00019476f
};

/* Delay line for anti-noise output history (used by Feedback Neutralizer) - Length: FB_FILTER_TAP (500) */
float Y_buf[FB_FILTER_TAP] = {0.0f};

/* Delay line for secondary path filtering - Length: SEC_FILTER_TAP (500) */
float X_hat[SEC_FILTER_TAP] = {0.0f};

/* Filtered-X delay line: x'(n), x'(n-1), ... - Length: FILTER_TAP (256) */
float Fx[FILTER_TAP] = {0.0f};

/* =========================================================================
 * Runtime Tuning and Control Variables (Accessible via CCS Expressions)
 * ========================================================================= */
volatile int   anc_enable             = 1;            /* 1: ANC Active, 0: ANC Bypassed / Muted */
volatile int   speaker_channel        = 0;            /* 0: Both L+R (Matches working NLMS0), 1: Right-Only, 2: Left-Only */
volatile int   test_tone_mode         = 0;            /* 1: Output test sound to verify speaker/amp wiring, 0: Normal ANC */
volatile int   inv_polarity           = 0;            /* 0: Standard inverted phase (-y), 1: Non-inverted (+y) */
volatile int   reset_filter           = 0;            /* Set to 1 in Expressions to instantly reset filter weights to 0 */
volatile float mu                     = 0.001f;       /* Step size for smooth, stable convergence without jitter */
volatile float leakage                = 1.0f;         /* 1.0f = Pure unconstrained LMS for maximum cancellation (23 dB) */
volatile float noise_gate_thresh      = 0.0f;         /* 0.0f: Disabled to prevent gate chattering */

/* Acoustic Feedback Neutralization Controls */
volatile int   feedback_cancel_enable = 1;            /* 1: Feedback Neutralizer ON, 0: Bypassed (original echo) */
volatile float fb_scale               = 1.0f;         /* 1.0 = 100% feedback echo subtraction */

/* Test sound PRNG state */
static uint32_t test_prng = 12345;

/* Real-time channel routing controls (Switchable in CCS Expressions) */
volatile int   ref_channel      = 1;            /* 1: Left is Ref, Right is Err / 2: Right is Ref, Left is Err */

/* Real-time CCS Graph Monitoring Buffers (Matches MIC_TEST & NLMS0) */
int16_t ref_mic_buf[MONITOR_BUFLEN]   = {0};      /* Raw 16-bit Reference mic ADC (Contains feedback echo) */
int16_t ref_clean_buf[MONITOR_BUFLEN] = {0};      /* Cleaned Reference mic ADC (Feedback echo REMOVED!) */
int16_t err_mic_buf[MONITOR_BUFLEN]   = {0};      /* Raw 16-bit Error mic ADC (Plot with: 16 bit signed integer) */
int16_t spk_out_buf[MONITOR_BUFLEN]   = {0};      /* Raw 16-bit Canceling speaker output (Plot with: 16 bit signed integer) */
float   ref_x_buf[MONITOR_BUFLEN]     = {0.0f};   /* Normalized float Reference mic x(n) (Plot with: 32 bit float) */
float   err_e_buf[MONITOR_BUFLEN]     = {0.0f};   /* Normalized float Error mic e(n) (Plot with: 32 bit float) */
float   anti_y_buf[MONITOR_BUFLEN]    = {0.0f};   /* Anti-noise float output y(n) (Plot with: 32 bit float) */
float   fx_buf[MONITOR_BUFLEN]        = {0.0f};   /* Filtered-X signal x'(n) */
static int mon_idx = 0;

/* Real-time debug scalar variables for CCS Expressions watch */
volatile float debug_x        = 0.0f;             /* Raw Reference mic input (x_n) */
volatile float debug_x_clean  = 0.0f;             /* Cleaned Reference mic input without echo */
volatile float debug_y_fb     = 0.0f;             /* Feedback echo signal being subtracted */
volatile float debug_e        = 0.0f;             /* Error mic input */
volatile float debug_y        = 0.0f;             /* Filter anti-noise output */
volatile float debug_fx       = 0.0f;             /* Filtered-X sample */
volatile float debug_err_power= 0.0f;             /* Running energy of error mic */
volatile float debug_ref_pwr  = 0.0f;             /* Running energy of ref mic */
volatile float debug_w_max    = 0.0f;             /* Peak weight magnitude in W(z) */

/* Real-time Performance & XRUN Diagnostics (Watch in CCS Expressions) */
volatile uint32_t isr_cycles     = 0;       /* Last ISR execution time in CPU cycles */
volatile uint32_t isr_max_cycles = 0;       /* Peak ISR execution time (budget: 18,750 @ 300MHz) */
volatile uint32_t xrun_cnt       = 0;       /* McASP Overrun/Underrun counter (0 = perfect real-time) */


/* =========================================================================
 * Data and Buffer Initialization
 * ========================================================================= */
void ClearData( void ) {
    int i;
    for(i = 0; i < FILTER_TAP; i++) {
        W[i]  = 0.0f;
        X[i]  = 0.0f;
        Fx[i] = 0.0f;
    }
    for(i = 0; i < SEC_FILTER_TAP; i++) {
        X_hat[i] = 0.0f;
    }
    for(i = 0; i < FB_FILTER_TAP; i++) {
        Y_buf[i] = 0.0f;
    }
    for(i = 0; i < MONITOR_BUFLEN; i++) {
        ref_mic_buf[i]   = 0;
        ref_clean_buf[i] = 0;
        err_mic_buf[i]   = 0;
        spk_out_buf[i]   = 0;
        ref_x_buf[i]     = 0.0f;
        err_e_buf[i]     = 0.0f;
        anti_y_buf[i]    = 0.0f;
        fx_buf[i]        = 0.0f;
    }
    mon_idx = 0;
    led_cnt = 0;
    debug_err_power = 0.0f;

    /* Turn on LED D6 to show ANC is active by default */
    if(anc_enable) {
        LED_On( LED_D6 );
    } else {
        LED_Off( LED_D6 );
    }
}



/* =========================================================================
 * McASP Audio Interrupt Service Routine (Real-Time FxNLMS)
 * Execution frequency: 16,000 Hz (every 62.5 microseconds)
 * ========================================================================= */
void MCASP_ISR( void ) {
    uint32_t t_start = TSCL;
    uint32_t rx_sample = mcaspRegs->RBUF14;
    int k;

    /* Check McASP Overrun / Underrun flags */
    if (mcaspRegs->RSTAT & 0x01) { xrun_cnt++; mcaspRegs->RSTAT = 0x01; }
    if (mcaspRegs->XSTAT & 0x01) { xrun_cnt++; mcaspRegs->XSTAT = 0x01; }

    /* 1. Receive 2-channel 16-bit audio data from Line-in */
    Int16 ch_left  = (Int16)(rx_sample >> 16);     /* Line-in Left (Tip) */
    Int16 ch_right = (Int16)(rx_sample & 0xFFFF);  /* Line-in Right (Ring) */

    /* 
     * 2. Channel mapping: split reference mic x(n) and error mic e(n)
     * ref_channel:
     *   1: Left = Reference Mic, Right = Error Mic (Default)
     *   2: Right = Reference Mic, Left = Error Mic (Swap if wired inversely)
     */
    Int16 raw_ref = (ref_channel == 2) ? ch_right : ch_left;
    Int16 raw_err = (ref_channel == 2) ? ch_left  : ch_right;

    /* 1st-order DC Blocker Filter (removes ADC DC bias < 15 Hz) */
    static float dc_x_in = 0.0f, dc_x_out = 0.0f;
    static float dc_e_in = 0.0f, dc_e_out = 0.0f;

    float raw_x_f = (float)raw_ref / 32768.0f;
    float x_n = raw_x_f - dc_x_in + 0.995f * dc_x_out;
    dc_x_in = raw_x_f;
    dc_x_out = x_n;

    float raw_e_f = (float)raw_err / 32768.0f;
    float e_n = raw_e_f - dc_e_in + 0.995f * dc_e_out;
    dc_e_in = raw_e_f;
    dc_e_out = e_n;

    /* 
     * 2-1. Acoustic Feedback Neutralization (Canceling Speaker -> Reference Mic)
     * Estimates speaker sound that arrived at Ref Mic and subtracts it out.
     */
    float y_fb = 0.0f;
    if(feedback_cancel_enable) {
        for(k = 0; k < FB_FILTER_TAP; k++) {
            y_fb += F_hat[k] * Y_buf[k];
        }
    }
    float x_clean = x_n - fb_scale * y_fb;

    /* Reference Signal Power Estimation & Noise Gate Detector (Monitors pure noise) */
    static float ref_pwr_est = 0.0f;
    ref_pwr_est = 0.99f * ref_pwr_est + 0.01f * (x_clean * x_clean);
    int sound_active = (noise_gate_thresh <= 0.0f) || (ref_pwr_est >= noise_gate_thresh);

    /* Record samples into 16-bit graph buffers (Matches MIC_TEST!) */
    ref_mic_buf[mon_idx]   = raw_ref;
    ref_clean_buf[mon_idx] = (Int16)(x_clean * 32767.0f);
    err_mic_buf[mon_idx]   = raw_err;

    /* 3. Update reference signal delay line: X[k] with pure noise x_clean */
    for(k = FILTER_TAP - 1; k > 0; k--) {
        X[k] = X[k - 1];
    }
    X[0] = x_clean;

    /* Check if runtime reset requested via CCS Expressions */
    if(reset_filter) {
        reset_filter = 0;
        for(k = 0; k < FILTER_TAP; k++) {
            W[k]  = 0.0f;
            X[k]  = 0.0f;
            Fx[k] = 0.0f;
        }
        for(k = 0; k < SEC_FILTER_TAP; k++) {
            X_hat[k] = 0.0f;
        }
        for(k = 0; k < FB_FILTER_TAP; k++) {
            Y_buf[k] = 0.0f;
        }
    }

    /* 4. Calculate primary filter output: y(n) = W^T * X */
    float y = 0.0f;
    if(anc_enable && sound_active) {
        for(k = 0; k < FILTER_TAP; k++) {
            y += W[k] * X[k];
        }
    }

    /* 5. Phase inversion for acoustic cancellation (Switchable via inv_polarity) */
    float y_out;
    if(inv_polarity == 0) {
        y_out = -y;   /* Standard inverted anti-noise */
    } else {
        y_out = y;    /* Inverted hardware polarity fallback */
    }

    /* Saturation and clipping protection */
    if (y_out > 1.0f)       y_out = 1.0f;
    else if (y_out < -1.0f) y_out = -1.0f;

    Int16 out_val = 0;
    if(test_tone_mode) {
        /* Direct speaker test mode: generate soft white noise (-12 dBFS) to verify hardware */
        test_prng = test_prng * 1664525 + 1013904223;
        out_val = (int16_t)(test_prng >> 16) / 4;
    } else if(sound_active) {
        out_val = (Int16)(y_out * 32767.0f);
    } else {
        out_val = 0;    /* External sound is OFF -> 100% mute speaker */
    }

    /* 6. Output to secondary speaker via McASP */
    uint16_t out_left  = 0x0000;
    uint16_t out_right = 0x0000;

    if(speaker_channel == 0) {
        out_left  = (uint16_t)out_val;  /* Output to Both Left & Right */
        out_right = (uint16_t)out_val;
    } else if(speaker_channel == 1) {
        out_left  = 0x0000;             /* Left strictly 0x0000 for TPA3116D2 protection */
        out_right = (uint16_t)out_val;  /* Right channel speaker */
    } else if(speaker_channel == 2) {
        out_left  = (uint16_t)out_val;  /* Left channel speaker */
        out_right = 0x0000;
    }

    mcaspRegs->XBUF13 = ((uint32_t)out_left << 16) | (uint32_t)out_right;

    /* Update anti-noise history buffer for feedback neutralizer */
    for(k = FB_FILTER_TAP - 1; k > 0; k--) {
        Y_buf[k] = Y_buf[k - 1];
    }
    Y_buf[0] = (float)out_val / 32768.0f;

    /* 7. Secondary Path Filtering: Calculate Filtered-X signal: x'(n) = S_hat * x_clean(n) */
    /* 7-1. Update secondary path delay buffer */
    for(k = SEC_FILTER_TAP - 1; k > 0; k--) {
        X_hat[k] = X_hat[k - 1];
    }
    X_hat[0] = x_clean;

    /* 7-2. S_hat FIR filtering (500 taps) */
    float fx_n = 0.0f;
    for(k = 0; k < SEC_FILTER_TAP; k++) {
        fx_n += S_hat[k] * X_hat[k];
    }

    /* 7-3. Update Filtered-X delay line: Fx[k] */
    for(k = FILTER_TAP - 1; k > 0; k--) {
        Fx[k] = Fx[k - 1];
    }
    Fx[0] = fx_n;

    /* 8. Normalized LMS (NLMS) Primary Weight Adaptation with Stability Safeguards */
    if(anc_enable && !test_tone_mode) {
        if(sound_active) {
            /* 8-1. Energy normalization with robust regularization (prevents near-zero divergence) */
            float norm_pwr = 0.0005f;
            for(k = 0; k < FILTER_TAP; k++) {
                norm_pwr += Fx[k] * Fx[k];
            }

            /* 8-2. Normalized step size with upper-bound safety limiter */
            float mu_norm = mu / norm_pwr;
            if(mu_norm > 2.0f) mu_norm = 2.0f;

            /* 8-3. Update primary filter weights with leakage, polarity switch, and tap clamping */
            float step = (inv_polarity == 0) ? (mu_norm * e_n) : (-mu_norm * e_n);
            for(k = 0; k < FILTER_TAP; k++) {
                float w_new = W[k] * leakage + step * Fx[k];
                if(w_new > 0.5f)       w_new = 0.5f;
                else if(w_new < -0.5f) w_new = -0.5f;
                W[k] = w_new;
            }
        } else {
            /* External sound is OFF: fast decay weights to silence speaker and stop howling */
            for(k = 0; k < FILTER_TAP; k++) {
                W[k] *= 0.99f;
            }
        }
    }

    /* Record active signals into waveform buffers for CCS Graph monitoring */
    spk_out_buf[mon_idx] = out_val;
    ref_x_buf[mon_idx]   = x_clean;
    err_e_buf[mon_idx]   = e_n;
    anti_y_buf[mon_idx]  = y;
    fx_buf[mon_idx]      = fx_n;
    if(++mon_idx >= MONITOR_BUFLEN) {
        mon_idx = 0;
    }

    /* Update real-time debug scalar variables */
    debug_x       = x_n;
    debug_x_clean = x_clean;
    debug_y_fb    = y_fb;
    debug_e       = e_n;
    debug_y       = y;
    debug_fx      = fx_n;
    debug_err_power = 0.999f * debug_err_power + 0.001f * (e_n * e_n);
    debug_ref_pwr   = ref_pwr_est;

    /* Update real-time performance diagnostics */
    uint32_t elapsed = TSCL - t_start;
    isr_cycles = elapsed;
    if (elapsed > isr_max_cycles) {
        isr_max_cycles = elapsed;
    }

    ++led_cnt;

}


/* =========================================================================
 * SYS/BIOS Hwi and Idle function definitions (matches app.cfg)
 * ========================================================================= */
void TIMER1_TINT12_ISR( void ) {
    StopTimer( (CSL_TmrRegsOvly)CSL_TMR_1_REGS );
    GPIO_ClearInterruptState( GP2 );
    ClearInterrupt( INT_NUM_GPIO_B2 );
    EnableInterrupt( INT_NUM_GPIO_B2 );
}

/* User Push Button ISR: Toggle ANC ON / OFF */
void GPIO_PUSHBUTTON_ISR( void ) {
    DisableInterrupt( INT_NUM_GPIO_B2 );
    StartTimer( (CSL_TmrRegsOvly)CSL_TMR_1_REGS );

    /* Toggle ANC enable state */
    anc_enable = !anc_enable;

    if(anc_enable) {
        LED_On( LED_D6 );   /* LED D6 ON = ANC Active */
    } else {
        LED_Off( LED_D6 );  /* LED D6 OFF = ANC Bypassed */
    }
}

/* Idle LED Task: Heartbeat on LED D4 & background monitoring */
void IdleLED( void ) {
    int k;
    float max_w = 0.0f;
    for(k = 0; k < FILTER_TAP; k++) {
        float abs_w = fabsf(W[k]);
        if(abs_w > max_w) max_w = abs_w;
    }
    debug_w_max = max_w;

    if( led_cnt >= (SAMPLING_FREQ >> 1) ) {
        LED_Toggle( LED_D4 );
        led_cnt = 0;
    }
}
