* By: Shayan Farhang Pazhooh
* Advisor: Dr. Hossein Shams Shemirani

* C: Number of CURRENT aircraft (already in hangar)
$set CVAL 02

* NVAL: Number of FUTURE aircraft
* TOTAL_N: Total number of aircraft (NVAL + CVAL)
* FVAL: Start index for FUTURE aircraft (CVAL + 1), pre-formatted and passed from command line.
* SAMPLE_ID: The identifier for the random sample instance.
* FILEPATH: The sub-directory path within 'data' (e.g., /random).
* FILENAME: The core filename segment (e.g., %TOTAL_N%-%SAMPLE_ID%).

* Set defaults for direct running of this single file (if not run from Python)
$if not set NVAL      $set NVAL 05
$if not set TOTAL_N   $set TOTAL_N 07
$if not set FVAL      $set FVAL 03
$if not set SAMPLE_ID $set SAMPLE_ID 01
$if not set FILEPATH  $set FILEPATH /random
$if not set FILENAME  $set FILENAME 07-01


Set
    m 'Aircraft models (numeric ids)' / 1*8/
    a 'All aircraft (both new and initial)' /a01*a%TOTAL_N%/
    c(a) 'Current aircraft (already in hangar)' /a01*a%CVAL%/
    f(a) 'Future aircraft (new arrivals)' /a%FVAL%*a%TOTAL_N%/;

Alias(a, b);
Alias(c, d);
Alias(f, g);


Scalars
    HW 'Width of the hangar' / 65 /
    HL 'Length of the hangar' / 60 /
    Buffer 'The buffer space between aircraft' / 5 /;

*epsilon
Scalars
    epsilon_t 'Minimum Inter-Arrival / Inter-Departure Time' / 0.1 /
    epsilon_p 'Positioning penalty coefficient' / 0.001 /;

Parameters
    W(a) 'Aircraft Width'
    L(a) 'Aircraft Length'
    ETA(a) 'Expected Time of Arrival'
    ETD(a) 'Expected Time of Departure'
    ServT(a) 'Service Time (or remaining for initial aircraft)'
    P_Rej(f) 'Penalty for rejecting a future aircraft'
    P_Arr(f) 'Penalty per unit of time for Arrival Delay'
    P_Dep(a) 'Penalty per unit of time for Departure Delay'
    X_init(c) 'Initial X position for aircraft in hangar'
    Y_init(c) 'Initial Y position for aircraft in hangar'
    M_ID(a) 'Model ID for each aircraft';


Parameters
    T1(m, *) 'Data for each aircraft Model'
    T2(c, *) 'Data for aircraft already in hangar (current)'
    T3(f, *) 'Data for new arrival aircraft (future)';

* Read data from the main 'data' folder for T1 and T2
$call gdxxrw i=data/T1.csv o=TEMP_T1.gdx par=T1 rng=A1 rdim=1 cdim=1
$call gdxxrw i=data/T2.csv o=TEMP_T2.gdx par=T2 rng=A1 rdim=1 cdim=1
* Read data for T3 using the dynamic path and filename
$call gdxxrw i=data%FILEPATH%/T3-%FILENAME%.csv o=TEMP_T3.gdx par=T3 rng=A1 rdim=1 cdim=1


$gdxin TEMP_T1.gdx
$load T1
$gdxin

$gdxin TEMP_T2.gdx
$load T2
$gdxin

$gdxin TEMP_T3.gdx
$load T3
$gdxin

*Aircraft Model
M_ID(c) = T2(c, 'M_ID');
M_ID(f) = T3(f, 'M_ID');

*lookup using sum on t that are = M_ID
W(a) = sum(m$(ord(m) = M_ID(a)), T1(m, 'W'));
L(a) = sum(m$(ord(m) = M_ID(a)), T1(m, 'L'));

*current aircraft data
ETD(c) = T2(c, 'ETD');
ServT(c) = T2(c, 'ServT');
X_init(c) = T2(c, 'Init_X');
Y_init(c) = T2(c, 'Init_Y');
P_Dep(c) = T2(c, 'P_Dep');

*future aircraft data
ETA(f) = T3(f, 'ETA');
ETD(f) = T3(f, 'ETD');
ServT(f) = T3(f, 'ServT');
P_Rej(f)  = T3(f, 'P_Rej');
P_Arr(f) = T3(f, 'P_Arr');
P_Dep(f) = T3(f, 'P_Dep');


*Big M
Scalars
    M_T 'Big M for Time-related constraints'
    M_X 'Big M for X-dimension spatial constraints'
    M_Y 'Big M for Y-dimension spatial constraints';

M_T = smax(f, ETA(f)) + sum(a, ServT(a));
M_X = HW;
M_Y = HL;

Variable Z;

Positive Variables
    X(a)
    Y(a)
    Roll_in(a) 'Actual time the aircraft rolls into its spot'
    Roll_out(a) 'Actual time the aircraft rolls out'
    D_Arr(f) 'Arrival Delay (Time an accepted future aircraft waits before roll-in)'
    D_Dep(a) 'Departure Delay (Lateness of departure)';

Binary Variables
    Accept(a) '1 if new aircraft f is accepted, 0 otherwise'
    Right(a, b) '1 if a is entirely to the right of b'
    Above(a, b) '1 if a is entirely above (larger Y) b'
    OutIn(a, b) '1 if a is out before b comes in'
    InIn(a, b) '1 if a comes in before b'
    OutOut(a, b) '1 if a is out before b'
    InOut(a, b) '1 if a comes in before b is out';

Equations
    e01 'Objective_Function'
    e02 'Acceptance_Rules'
    e03 'Min_Roll_in_Time'
    e04 'Service_Time_Check'
    e05 'Calculate_D_Arr'
    e06 'Calculate_D_Dep'
    e07 'Boundary_X_Min'
    e08 'Boundary_X_Max'
    e09 'Boundary_Y_Min'
    e10 'Boundary_Y_Max'
    e11 'Enforce_Position_Right'
    e12 'Enforce_Position_Above'
    e13 'Choose_Rel'
    e14 'Enforce_Time_Separation'
    e15 'Define_InIn_ff_1'
    e16 'Define_InIn_ff_2'
    e17 'Time Separation for Roll-out vs Roll-out'
    e18 'Time Separation for Roll-out vs Roll-out'
    e19 'Time Separation for Roll-in vs Roll-out'
    e20 'Time Separation for Roll-in vs Roll-out'
    e21 'Blocking_Rule_1'
    e22 'Blocking_Rule_2';



*Objective_Function
e01.. Z =e= sum(f, P_Rej(f) * (1 - Accept(f)))
    + sum(f, P_Arr(f) * D_Arr(f))
    + sum(a, P_Dep(a) * D_Dep(a))
    + epsilon_p * sum(f, X(f) + Y(f));


*Acceptance_Rules
e02(f).. X(f) + Y(f) + Roll_in(f) + Roll_out(f) + D_Arr(f) + D_Dep(f)
    =l= (M_X + M_Y + 4*M_T) * Accept(f);

*Min_Roll_in_Time
e03(f).. Roll_in(f) =g= ETA(f) * Accept(f);

*Service_Time_Check
e04(a).. Roll_out(a) - Roll_in(a) =g= ServT(a) * Accept(a);

*Calculate_D_Arr
e05(f).. D_Arr(f) =g= Roll_in(f) - ETA(f);

*Calculate_D_Dep
e06(a).. D_Dep(a) =g= Roll_out(a) - ETD(a);

*Boundary_X_Min
e07(f).. X(f) =g= Buffer * Accept(f);

*Boundary_X_Max
e08(f).. X(f) + W(f) =l= HW - Buffer + M_X * (1 - Accept(f));

*Boundary_Y_Min
e09(f).. Y(f) =g= Buffer * Accept(f);

*Boundary_Y_Max
e10(f).. Y(f) + L(f) =l= HL - Buffer + M_Y * (1 - Accept(f));

*Enforce_Position_Right
e11(a, b)$(ord(a) <> ord(b))..
    X(b) + W(b) + Buffer =l= X(a) + M_X * (1 - Right(a, b));

*Enforce_Position_Above
e12(a, b)$(ord(a) <> ord(b))..
    Y(b) + L(b) + Buffer =l= Y(a) + M_Y * (1 - Above(a, b));

*Choose_Rel
e13(a, b)$(ord(a) < ord(b))..
    Right(b, a) + Right(a, b) + Above(b, a) + Above(a, b)
    + OutIn(a, b) + OutIn(b, a)
    =g= Accept(a) + Accept(b) - 1;

*Enforce_Time_Separation
e14(a, b)$(ord(a) <> ord(b))..
    Roll_out(a) + epsilon_t =l= Roll_in(b) + M_T * (1 - OutIn(a, b));

*Define_InIn_ff_1
e15(f, g)$(ord(f) <> ord(g))..
    Roll_in(g) =g= Roll_in(f) + epsilon_t - M_T * (1 - InIn(f, g))
    - M_T * (2 - Accept(f) - Accept(g));

*Define_InIn_ff_2
e16(f, g)$(ord(f) <> ord(g))..
    Roll_in(f) =g= Roll_in(g) + epsilon_t - M_T * InIn(f, g)
    - M_T * (2 - Accept(f) - Accept(g));

*For any two accepted aircraft a and b, one must roll out at least epsilon_t before the other.
e17(a, b)$(ord(a) < ord(b))..
    Roll_out(b) =g= Roll_out(a) + epsilon_t - M_T * (1 - OutOut(a, b))
    - M_T * (2 - Accept(a) - Accept(b));

e18(a, b)$(ord(a) < ord(b))..
    Roll_out(a) =g= Roll_out(b) + epsilon_t - M_T * OutOut(a, b)
    - M_T * (2 - Accept(a) - Accept(b));

*For any two accepted aircraft a and b, the roll_in of one cannot be at the same time
e19(a, b)$((ord(a) <> ord(b)) and (not c(a) or not c(b)))..
    Roll_out(b) =g= Roll_in(a) + epsilon_t - M_T * (1 - InOut(a, b))
    - M_T * (2 - Accept(a) - Accept(b));

e20(a, b)$((ord(a) <> ord(b)) and (not c(a) or not c(b)))..
    Roll_in(a) =g= Roll_out(b) + epsilon_t - M_T * InOut(a, b)
    - M_T * (2 - Accept(a) - Accept(b));

*Blocking_Rule_1
e21(a, b)$((ord(a) <> ord(b)))..
    Roll_out(a) =g= Roll_out(b) + epsilon_t
    - M_T * ( (1 - Above(b, a)) + Right(a, b) + Right(b, a)
    + (1 - InIn(a, b)) );

*Blocking_Rule_2
e22(a, b)$((ord(a) <> ord(b)) and (not c(a) or not c(b)))..
    Roll_in(a) =g= Roll_out(b) + epsilon_t
    - M_T * ( (1 - Above(b, a)) + Right(a, b) + Right(b, a)
    + InIn(a, b) );



*For aircraft already in the hangar:
Accept.fx(c) = 1;
X.fx(c) = X_init(c);
Y.fx(c) = Y_init(c);
Roll_in.fx(c) = 0;

*Required for Blocking_Rule constraints to not add epsilon_t to the first plane Roll_in
InIn.fx(c, f) = 1;

*Symmetry:
InIn.fx(f, c) = 0;

*c, d form for Blocking_Rule_1
InIn.fx(c, d)$(ord(c) <> ord(d)) = 1;



Model HangarScheduling /all/;

option MIP = Cplex;
option solprint=off;
* These options are set via command line: reslim (time limit) and optcr (optimality gap)
* option reslim = 86400;
* option optcr = 0.00;
option limcol=0, limrow=100, solprint=off;
option threads = 0;

Solve HangarScheduling using MIP minimizing Z;

Scalar TotalCost_Reject, TotalCost_D_Arr, TotalCost_D_Dep;
TotalCost_Reject  = sum(f, P_Rej(f) * (1 - Accept.l(f)));
TotalCost_D_Arr = sum(f, P_Arr(f) * D_Arr.l(f));
TotalCost_D_Dep = sum(a, P_Dep(a) * D_Dep.l(a));

*Report:
Parameter SolutionReport(a,*) 'Report as a table';

SolutionReport(a, "Width") = W(a);
SolutionReport(a, "Length") = L(a);
SolutionReport(a, "ETA") = ETA(a);
SolutionReport(a, "ServT") = ServT(a);
SolutionReport(a, "ETD") = ETD(a);
SolutionReport(f, "P_Rej") = P_Rej(f);
SolutionReport(f, "P_Arr") = P_Arr(f);
SolutionReport(a, "P_Dep") = P_Dep(a);

SolutionReport(a, "Accepted") = Accept.L(a);
SolutionReport(a, "X") = X.L(a);
SolutionReport(a, "Y") = Y.L(a);

SolutionReport(a, "Roll_In") = Roll_in.L(a);
SolutionReport(a, "Roll_Out") = Roll_out.L(a);

SolutionReport(f, "D_Arr") = D_Arr.L(f);
SolutionReport(a, "D_Dep") = D_Dep.L(a);


Display Z.L, TotalCost_Reject, TotalCost_D_Arr, TotalCost_D_Dep,
        SolutionReport;

* --- CSV Export ---
* The output filename now includes both NVAL and SAMPLE_ID to be unique.
File reportfile /SolutionReport_N%NVAL%_S%SAMPLE_ID%.csv/;
put reportfile;

* csv header
put 'Aircraft_ID,',
    'Accepted,',
    'Width,',
    'Length,',
    'ETA,',
    'Roll_In,',
    'X,',
    'Y,',
    'ServT,',
    'ETD,',
    'Roll_Out,',
    'D_Arr,',
    'D_Dep,',
    'Penalty_Reject,',
    'Penalty_ArrivalDelay,',
    'Penalty_DepartureDelay,',
    'Hangar_Width,',
    'Hangar_Length,',
    'StartDate'
    /;

loop(a,
    put a.tl:0, ',',
        SolutionReport(a, "Accepted"):0:0, ',',
        W(a):0:0, ',',
        L(a):0:0, ',',
        ETA(a):0:2, ',',
        Roll_in.L(a):0:2, ',',
        X.L(a):0:2, ',',
        Y.L(a):0:2, ',',
        ServT(a):0:2, ',',
        ETD(a):0:2, ',',
        Roll_out.L(a):0:2, ',',
        SolutionReport(a, "D_Arr"):0:2, ',',
        SolutionReport(a, "D_Dep"):0:2, ',',
        SolutionReport(a, "P_Rej"):0:0, ',',
        SolutionReport(a, "P_Arr"):0:0, ',',
        SolutionReport(a, "P_Dep"):0:0, ',',
        HW:0:0, ',',
        HL:0:0, ',',
        "2025-07-15 11:00"
        /;
);
putclose reportfile;
