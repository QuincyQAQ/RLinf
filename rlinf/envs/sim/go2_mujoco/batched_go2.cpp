// CPU MuJoCo stepping for the Ascend learner. Each environment owns its model/data.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace py=pybind11;
using Array=py::array_t<double,py::array::c_style|py::array::forcecast>;
struct Env {
 mjModel* m=nullptr; mjData* d=nullptr; std::mt19937 rng;
 std::array<double,4> air{},stance{},lastair{},laststance{};
 std::vector<double> history; int cursor=0;
 Env(int seed):rng(seed){}
};
class Batch {
 std::vector<Env> envs; int n,threads,base; std::array<int,12> jids{},qadr{},vadr{},aids{};
 std::array<int,4> footgeom{},footbody{}; std::array<double,12> defaults={.1,-.1,.1,-.1,.8,.8,1.,1.,-1.5,-1.5,-1.5,-1.5};
 double uniform(Env& e,double lo,double hi){return std::uniform_real_distribution<double>(lo,hi)(e.rng);}
 void reset_one(Env& e){
  mj_resetData(e.m,e.d); auto*d=e.d; d->qpos[0]=uniform(e,-.5,.5);d->qpos[1]=uniform(e,-.5,.5);d->qpos[2]=.4;
  double yaw=uniform(e,-3.14,3.14);d->qpos[3]=cos(yaw/2);d->qpos[6]=sin(yaw/2);
  for(int j=0;j<12;++j){d->qpos[qadr[j]]=defaults[j];d->qvel[vadr[j]]=uniform(e,-1,1);}
  e.air.fill(0);e.stance.fill(0);e.lastair.fill(0);e.laststance.fill(0);std::fill(e.history.begin(),e.history.end(),0);e.cursor=0;
  mj_forward(e.m,d);
 }
 void contacts(Env&e){
  auto*m=e.m;auto*d=e.d;std::vector<double> forces(m->nbody*3,0);std::array<double,12> feet{};
  for(int c=0;c<d->ncon;++c){
   const auto& con=d->contact[c];double f[6],world[3];mj_contactForce(m,d,c,f);mju_mulMatTVec(world,con.frame,f,3,3);
   for(int side=0;side<2;++side){int g=con.geom[side];if(g<0)continue;double sign=side==0?-1:1;int leg=-1;
    for(int k=0;k<4;++k)if(g==footgeom[k])leg=k;
    int body=leg>=0?footbody[leg]:m->geom_bodyid[g];
    for(int k=0;k<3;++k){forces[body*3+k]+=sign*world[k];if(leg>=0)feet[leg*3+k]+=sign*world[k];}
   }
  }
  for(int b=0;b<m->nbody;++b){double norm=mju_norm3(&forces[3*b]);e.history[e.cursor*m->nbody+b]=norm;}
  for(int k=0;k<4;++k){bool contact=mju_norm3(&feet[k*3])>1.;
   if(contact){if(e.air[k]>0)e.lastair[k]=e.air[k];e.air[k]=0;e.stance[k]+=.005;}
   else{if(e.stance[k]>0)e.laststance[k]=e.stance[k];e.stance[k]=0;e.air[k]+=.005;}
  }
  e.cursor=(e.cursor+1)%3;
 }
 void observe(Env&e,double*out,const std::array<double,12>&tau,const std::array<double,12>&acc){
  auto*m=e.m;auto*d=e.d;
  std::copy(d->qpos,d->qpos+7,out);double vel[6];mj_objectVelocity(m,d,mjOBJ_BODY,base,vel,0);
  double rot[9];mju_quat2Mat(rot,d->qpos+3);mju_mulMatTVec(out+7,rot,vel+3,3,3);mju_mulMatTVec(out+10,rot,vel,3,3);
  double gravity[3]={0,0,-1};mju_mulMatTVec(out+13,rot,gravity,3,3);
  for(int j=0;j<12;++j){out[16+j]=d->qpos[qadr[j]];out[28+j]=d->qvel[vadr[j]];out[40+j]=tau[j];out[52+j]=acc[j];}
  for(int k=0;k<4;++k){double peak=0;for(int h=0;h<3;++h)peak=std::max(peak,e.history[h*m->nbody+footbody[k]]);
   out[64+k]=peak;mj_objectVelocity(m,d,mjOBJ_BODY,footbody[k],vel,0);std::copy(vel+3,vel+6,out+68+3*k);
   out[82+k]=e.lastair[k];out[86+k]=e.laststance[k];out[90+k]=(e.stance[k]>0&&e.stance[k]<.0200001)?1.:0.;
  }
  out[80]=0;out[81]=0;
  for(int b=1;b<m->nbody;++b){double peak=0;for(int h=0;h<3;++h)peak=std::max(peak,e.history[h*m->nbody+b]);
   if(b==base){out[81]=peak;continue;}bool foot=false;for(int f:footbody)if(f==b)foot=true;if(!foot&&peak>1)out[80]+=1;
  }
 }
public:
 Batch(std::string xml,int count,int workers,int seed,bool randomize):n(count),threads(workers){
  char error[2048];mjModel*original=mj_loadXML(xml.c_str(),nullptr,error,sizeof(error));if(!original)throw std::runtime_error(error);
  if(original->nq!=19||original->nv!=18||original->nu!=12){mj_deleteModel(original);throw std::runtime_error("Unexpected Go2 dimensions");}
  base=mj_name2id(original,mjOBJ_BODY,"base_link");std::array<std::string,4> legs={"FL","FR","RL","RR"};
  for(int k=0;k<4;++k){footgeom[k]=mj_name2id(original,mjOBJ_GEOM,legs[k].c_str());footbody[k]=mj_name2id(original,mjOBJ_BODY,(legs[k]+"_foot").c_str());if(footgeom[k]<0||footbody[k]<0)throw std::runtime_error("Missing foot");}
  std::array<std::string,3> joints={"hip","thigh","calf"};
  for(int t=0;t<3;++t)for(int l=0;l<4;++l){int j=t*4+l;std::string name=legs[l]+"_"+joints[t];jids[j]=mj_name2id(original,mjOBJ_JOINT,(name+"_joint").c_str());aids[j]=mj_name2id(original,mjOBJ_ACTUATOR,name.c_str());if(jids[j]<0||aids[j]<0)throw std::runtime_error("Missing joint");qadr[j]=original->jnt_qposadr[jids[j]];vadr[j]=original->jnt_dofadr[jids[j]];}
  envs.reserve(n);for(int i=0;i<n;++i){envs.emplace_back(seed+i*7919);auto&e=envs.back();e.m=mj_copyModel(nullptr,original);e.d=mj_makeData(e.m);e.history.resize(3*e.m->nbody,0);
   if(randomize){double delta=uniform(e,-1,3),mass=e.m->body_mass[base];e.m->body_mass[base]+=delta;for(int a=0;a<3;++a)e.m->body_inertia[base*3+a]*=(mass+delta)/mass;
    for(int g=0;g<e.m->ngeom;++g)if(e.m->geom_bodyid[g]>0){double mu=.3+std::floor(uniform(e,0,64))*.9/63.;e.m->geom_friction[3*g]=mu;}
    mj_setConst(e.m,e.d);
   }reset_one(e);
  }mj_deleteModel(original);
 }
 ~Batch(){for(auto&e:envs){mj_deleteData(e.d);mj_deleteModel(e.m);}}
 py::array_t<double> step(Array actions){
  if(actions.ndim()!=2||actions.shape(0)!=n||actions.shape(1)!=12)throw std::runtime_error("Actions must be N x 12");
  const double*act=actions.data();py::array_t<double> result({n,94});auto*out=result.mutable_data();
  {py::gil_scoped_release release;
   #pragma omp parallel for num_threads(threads) schedule(static)
   for(int i=0;i<n;++i){auto&e=envs[i];auto*m=e.m;auto*d=e.d;std::array<double,12> tau{},acc{};
    for(int sub=0;sub<4;++sub){std::array<double,12> oldvel{};
     for(int j=0;j<12;++j){double v=d->qvel[vadr[j]];oldvel[j]=v;double target=std::clamp(defaults[j]+.25*act[i*12+j],-100.,100.);double effort=25*(target-d->qpos[qadr[j]])-.5*v;
      double lim=v*effort>0?20.2:23.4;if(std::abs(v)>=13.5)lim=std::max(0.,lim*(30-std::abs(v))/(30-13.5));tau[j]=std::clamp(effort,-lim,lim);d->ctrl[aids[j]]=tau[j];}
     mj_step(m,d);contacts(e);for(int j=0;j<12;++j)acc[j]=(d->qvel[vadr[j]]-oldvel[j])/.005;
    }mj_forward(m,d);observe(e,out+i*94,tau,acc);
   }
  }return result;
 }
 py::array_t<double> reset(py::array_t<int,py::array::c_style|py::array::forcecast> ids){
  py::array_t<double> result({int(ids.size()),94});auto*out=result.mutable_data();auto*indices=ids.data();int count=ids.size();
  for(int j=0;j<count;++j)if(indices[j]<0||indices[j]>=n)throw std::runtime_error("Bad environment id");
  {py::gil_scoped_release release;
   #pragma omp parallel for num_threads(threads) schedule(static)
   for(int k=0;k<count;++k){auto&e=envs[indices[k]];reset_one(e);std::array<double,12> zero{};observe(e,out+k*94,zero,zero);}
  }return result;
 }
 void push(Array velocities){if(velocities.ndim()!=2||velocities.shape(0)!=n||velocities.shape(1)!=3)throw std::runtime_error("Bad push array");auto v=velocities.unchecked<2>();for(int i=0;i<n;++i)if(v(i,2)>0){envs[i].d->qvel[0]=v(i,0);envs[i].d->qvel[1]=v(i,1);}}
 py::array_t<double> limits(){py::array_t<double> r({12,2});auto a=r.mutable_unchecked<2>();auto*m=envs[0].m;for(int j=0;j<12;++j){double lo=m->jnt_range[2*jids[j]],hi=m->jnt_range[2*jids[j]+1],mid=(lo+hi)/2,half=(hi-lo)*.45;a(j,0)=mid-half;a(j,1)=mid+half;}return r;}
 py::array_t<double> state(){py::array_t<double> r({n,38});auto*p=r.mutable_data();for(int i=0;i<n;++i){auto*d=envs[i].d;p[i*38]=d->time;std::copy(d->qpos,d->qpos+19,p+i*38+1);std::copy(d->qvel,d->qvel+18,p+i*38+20);}return r;}
};
PYBIND11_MODULE(batched_go2,m){py::class_<Batch>(m,"Batch").def(py::init<std::string,int,int,int,bool>()).def("step",&Batch::step).def("reset",&Batch::reset).def("push",&Batch::push).def("limits",&Batch::limits).def("state",&Batch::state);}
