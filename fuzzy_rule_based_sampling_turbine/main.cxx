
#include <cpp_headers.h>

#include <glm_headers.h>

#include <utils.h> // Contains fuzzy evaluation functions

#include <vtk_headers.h>

#include <sys/time.h>

using namespace std;

char* DATA_PATH = "/home/soumya/Desktop/33101";

int BIN=10;
int numBlocks =36;
int dim[3] = {151,71,56};
int startT=33101; //TODO
int numSteps = 1; //TODO
int finalT=startT+25*numSteps;

double clkbegin=0;
double clkend=0;
double writeclkbegin=0;
double writeclkend=0;
double writeTime=0;
double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

//Definition of methods
int compute_3d_to_1d_map(int x,int y,int z, int dimx, int dimy, int dimz)
{
    
    return x + dimx*(y+dimy*z);
}

void write_multiblock(vtkSmartPointer<vtkMultiBlockDataSet> mb,int timestep)
{
    stringstream ss;
    ss<<timestep;
    string fname = "/home/soumya/Desktop/out_" + ss.str() + ".vtm";
    //string fname = "out_" + ss.str() + ".vtm";
    vtkSmartPointer<vtkXMLMultiBlockDataWriter> mbwriter = vtkSmartPointer<vtkXMLMultiBlockDataWriter>::New();
    mbwriter->SetInputData(mb);
    mbwriter->SetEncodeAppendedData(0);
    mbwriter->SetFileName(fname.c_str());
    mbwriter->Write();
}

void  load_list(string list_fname,vtkSmartPointer<vtkMultiBlockDataSet> mb)
{
    FILE *fp = fopen(list_fname.c_str(), "rt");
    char s[1024];
    fgets(s, 1024, fp);
    int blocks = atoi(s);
    int i;

    vector<string> qfiles, gfiles;
    for (int i=0; i<blocks; i++)
    {
        char file1[1024], file2[1024]; // q, xyz
        fgets(s, 1024, fp);
        *strchr(s, '\n')=0; // remove the last new-line
        sprintf(file1, "%s/%s", DATA_PATH, s);
        fgets(s, 1024, fp);
        *strchr(s, '\n')=0; // remove the last new-line
        sprintf(file2, "%s/%s", DATA_PATH, s);
        //prints loaded file names
        //printf("xyz: [%s]   q: [%s]\n", file1, file2);

        gfiles.push_back(string(file1));
        qfiles.push_back(string(file2));
    }
    fclose(fp);

    //#pragma omp parallel for
    for (i=0; i<blocks; i++)
    {
        // Start by loading some data.
        vtkSmartPointer<vtkMultiBlockPLOT3DReader> reader = vtkSmartPointer<vtkMultiBlockPLOT3DReader>::New();
        reader->SetXYZFileName(gfiles[i].c_str());
        reader->SetQFileName(qfiles[i].c_str());
        reader->SetAutoDetectFormat(1);

        //reader->AddFunction(100); //density
        reader->AddFunction(110); //pressure
        reader->AddFunction(120); //temp
        //reader->AddFunction(130); //enthalpy
        //reader->AddFunction(140); //internal energy
        //reader->AddFunction(144); //kinetic energy
        //reader->AddFunction(153); //vel magnitude
        //reader->AddFunction(163); //stagnation energy
        reader->AddFunction(170); //entropy
        //reader->AddFunction(184); //swirl
        //reader->AddFunction(211); //vorticity magnitude

        //available vector fields in the data
        reader->AddFunction(200); //velocity
        //reader->AddFunction(201); //vorticity
        //reader->AddFunction(202); //momentum
        //reader->AddFunction(210); //pressure gradient
        //reader->AddFunction(212); //starin rate

        reader->Update();
        vtkDataSet *current_data = vtkDataSet::SafeDownCast(reader->GetOutput()->GetBlock(0));

        //extract uvel
        ////////////////////////////////////////////////////////////////////////////////
        vtkSmartPointer<vtkFloatArray> uvel = vtkSmartPointer<vtkFloatArray>::New();
        uvel->SetName("Uvel");

        vtkDataArray* velocity_array;
        vtkPointData* PointData;
        PointData = current_data->GetPointData();
        velocity_array = PointData->GetArray("Velocity");
        uvel->Resize(velocity_array->GetSize()/3);
        for(int p=0;p<velocity_array->GetSize()/3;p++)
        {
            double value[3];
            velocity_array->GetTuple(p,value);
            float datavalue = value[0];
            uvel->InsertTuple1(p,datavalue);
        }
        current_data->GetPointData()->AddArray(uvel);

        //#pragma omp critical
        //Add block to multiblock data
        mb->SetBlock(i, current_data);
    }

    gfiles.clear();
    qfiles.clear();
}

int main(int argc, char** argv)
{
    /* initialize random seed: */
    srand (time(NULL));

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // Initialize some data structures which will be needed for fuzzy based tracking
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    const int rule_num = 3; //TODO
    const int input_dim_num = 3; //TODO

    float temp_rule_matrix1[rule_num][input_dim_num];
    float temp_rule_matrix2[rule_num][input_dim_num];
    string line;
    vector<glm::vec2> temp_rule_matrix;

    //Read the parameters of the trained fuzzy rule based system
    ///////////////////////////////////////////////////////////////////
    ifstream readoutFis;
    readoutFis.open("/home/soumya/Desktop/outputmfs.txt");

    //Read outmfs
    getline(readoutFis, line);
    vector<float> outparamvals = split(line, ",");

    //Read inmfs
    ifstream readinFis;
    readinFis.open("/home/soumya/Desktop/inputmfs.txt");

    while(!readinFis.eof())
    {
        getline(readinFis, line);

        if(line[0]!=NULL) //to deal with the last empty line, basically this helps to ignore it
        {
            vector<float> v = split(line, ",");

            if(v.size()>0)
            {
                temp_rule_matrix.push_back(glm::vec2(v[0],v[1]));
            }
        }
    }

    int ij=0;
    for(int qq=0;qq<input_dim_num;qq++)
    {
        for(int jj=0;jj<rule_num;jj++)
        {
            glm::vec2 v21 = temp_rule_matrix[ij++];
            temp_rule_matrix1[jj][qq] = v21.x; // sigma vals
            temp_rule_matrix2[jj][qq] = v21.y; // mean vals
        }
    }

    temp_rule_matrix.clear();

    ///////////////////////////////////////////////
    //Fuzzy rule based inference system creation
    ///////////////////////////////////////////////
    Rule_Based_System rulebase;
    rulebase.num_rules = rule_num;
    rulebase.num_input_dim = input_dim_num;
    rulebase.fuzzy_system_type = "TSK";

    for(int qq=0;qq<rule_num;qq++)
    {
        Rules rule;
        rule.membership_func_type = "GMF";
        
        for(int jj=0;jj<input_dim_num;jj++)
        {
            Membership_func mm;
            mm.sigma = temp_rule_matrix1[qq][jj]; // sigma vals
            mm.mean = temp_rule_matrix2[qq][jj]; // mean vals
            rule.inputmfs.push_back(mm);
        }

        for(int jj=0;jj<=input_dim_num;jj++)
        {
            rule.out_params.push_back(outparamvals[jj]);
        }

        rulebase.rules.push_back(rule);
    }

    //////////////////////////////////////////////////////////////////////////////////
    //At this point the rule based system is ready to use
    //////////////////////////////////////////////////////////////////////////////////
    //omp_set_num_threads(256);

    string ffname = "timing.txt";
    ofstream globfptr;
    globfptr.open(ffname.c_str(),ios::out);
    float *datavals;
    datavals = (float *)malloc(sizeof(float)*input_dim_num);

    for(int ii=startT;ii<finalT;ii=ii+25)
    {
        cout<<endl<<"processing timestep: "<<ii<<endl;
        
        //Now load data here for inference
        //////////////////////////////////////////////////////
        vtkSmartPointer<vtkMultiBlockDataSet> mb_stallness = vtkSmartPointer<vtkMultiBlockDataSet>::New();

        vtkSmartPointer<vtkMultiBlockDataSet> mb = vtkSmartPointer<vtkMultiBlockDataSet>::New();
        
        stringstream ss;
        ss<<ii; //set the timestep to load
        string list_file_path = "/home/soumya/Dropbox/Codes/Pvis18_fuzzy_insitu_codes/evaluate_rulebase_turbine/list_files_14.2_all/";
        string filename = list_file_path + ss.str() + ".list";

        //Loads the plot3d data into a vtk multiblock dataset
        clkbegin = rtclock();
        load_list(filename,mb);
        clkend = rtclock();
        globfptr<<clkend-clkbegin<<" ";
        cout<<"data loading time: "<<clkend-clkbegin<<" secs"<<endl;


        int *Histogram;
        Histogram = (int *)malloc(BIN*sizeof(int));
        for(int kk=0;kk<BIN;kk++)
        {
            Histogram[kk] = 0;
        }

        
        clkbegin = rtclock();
        for(int bb=0; bb<mb->GetNumberOfBlocks();bb++)
        {
            vtkSmartPointer<vtkDataSet> current_passage = vtkDataSet::SafeDownCast(mb->GetBlock(bb));

            vtkSmartPointer<vtkPolyData> new_block = vtkSmartPointer<vtkPolyData>::New();
            vtkSmartPointer<vtkPoints> newPoints = vtkSmartPointer<vtkPoints>::New();

            //create stallness array
            vtkSmartPointer<vtkFloatArray> stallness = vtkSmartPointer<vtkFloatArray>::New();
            stallness->SetName("Stallness");
            stallness->SetNumberOfComponents(1);

            int index=0;
            int id=0;
            for(int r=0;r<dim[2];r++)
                for(int q=0;q<dim[1];q++)
                    for(int p=0;p<dim[0];p++)
                    {
                        index = compute_3d_to_1d_map(p,q,r,dim[0],dim[1],dim[2]);
                        float uvel_val = current_passage->GetPointData()->GetArray("Uvel")->GetTuple1(index);
                        float entropy_val = current_passage->GetPointData()->GetArray("Entropy")->GetTuple1(index);
                        float temperature_val = current_passage->GetPointData()->GetArray("Temperature")->GetTuple1(index);

                        datavals[0] = uvel_val;
                        datavals[1] = entropy_val;
                        datavals[2] = temperature_val;

                        float rett = evaluate_rulebase(rulebase,datavals,input_dim_num);

                        if(rett>1.0) //clamp to 1 for now
                            rett=1.0;
                        else if(rett<0)//clamp to 0 for now
                            rett=0;

                        //fuzzy value based sample selection
                        /*double random_val = ((double) rand() / (RAND_MAX));

                        if(random_val>0.9)
                        {
                            newPoints->InsertNextPoint(current_passage->GetPoint(id));
                            stallness->InsertNextTuple1(rett);
                        }*/

                        int bin_id = (int)(rett*(BIN-1));
                        Histogram[bin_id]++;

                        id++;
                    }

            new_block->SetPoints(newPoints.GetPointer());
            new_block->GetPointData()->AddArray(stallness.GetPointer());
            mb_stallness->SetBlock(bb,new_block.GetPointer());
        }

        for(int kk=0;kk<BIN;kk++)
        {
            cout<<Histogram[kk]<<endl;
        }

        clkend = rtclock();
        globfptr<<clkend-clkbegin<<endl;
        cout<<"prediction time: "<<clkend-clkbegin<<" secs"<<endl;

        ///write file out with inference information
        /////////////////////////////////////////////
        clkbegin = rtclock();
        write_multiblock(mb_stallness,ii);
        clkend = rtclock();
        globfptr<<clkend-clkbegin<<endl;
    }

    globfptr.close();

    return 0;
}
