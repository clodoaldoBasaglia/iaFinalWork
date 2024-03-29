from PIL import Image
import cv2
import numpy as np
import pandas as pd
from sklearn import preprocessing

import os.path

def extrairCaracteristicas(img):
    imagem = cv2.imread(img)
    blur = cv2.blur(imagem,(5,5))
    edge = cv2.Canny(blur,0,100,3)
    return edge.flatten()

def resize(caminho,arq):
    imagem = Image.open(caminho)
    altura, largura= 20,20
    saida = imagem.resize((largura,altura),Image.NEAREST)
    pasta = arq[0:3]
    if not os.path.exists('treinoResize'+pasta):
        os.mkdir('treinoResize'+pasta)
    caminhoSaida = 'treinoResize'+arq
    saida.save(caminhoSaida)
    return caminhoSaida

def main():
    separator = os.sep
    treino = open('NCharacter_SD19_BMP'+separator+'NIST_Train_Upper.txt','r')
    arqTreinos = treino.readlines()
    cont = 0
    treinoFinal = open('treinoFinal.csv','w',newline='\n')
    le = preprocessing.LabelEncoder()
    treinoFinal.write('c1;c2;c3;c4;c5;c6;c7;c8;c9;c10;c11;c12;c13;c14;c15;c16;c17;c18;c19;c20;c21;c22;c23;c24;c25;c26;c27;c28;c29;c30;c31;c32;c33;c34;c35;c36;c37;c38;c39;c40;c41;c42;c43;c44;c45;c46;c47;c48;c49;c50;c51;c52;c53;c54;c55;c56;c57;c58;c59;c60;c61;c62;c63;c64;c65;c66;c67;c68;c69;c70;c71;c72;c73;c74;c75;c76;c77;c78;c79;c80;c81;c82;c83;c84;c85;c86;c87;c88;c89;c90;c91;c92;c93;c94;c95;c96;c97;c98;c99;c100;c101;c102;c103;c104;c105;c106;c107;c108;c109;c110;c111;c112;c113;c114;c115;c116;c117;c118;c119;c120;c121;c122;c123;c124;c125;c126;c127;c128;c129;c130;c131;c132;c133;c134;c135;c136;c137;c138;c139;c140;c141;c142;c143;c144;c145;c146;c147;c148;c149;c150;c151;c152;c153;c154;c155;c156;c157;c158;c159;c160;c161;c162;c163;c164;c165;c166;c167;c168;c169;c170;c171;c172;c173;c174;c175;c176;c177;c178;c179;c180;c181;c182;c183;c184;c185;c186;c187;c188;c189;c190;c191;c192;c193;c194;c195;c196;c197;c198;c199;c200;c201;c202;c203;c204;c205;c206;c207;c208;c209;c210;c211;c212;c213;c214;c215;c216;c217;c218;c219;c220;c221;c222;c223;c224;c225;c226;c227;c228;c229;c230;c231;c232;c233;c234;c235;c236;c237;c238;c239;c240;c241;c242;c243;c244;c245;c246;c247;c248;c249;c250;c251;c252;c253;c254;c255;c256;c257;c258;c259;c260;c261;c262;c263;c264;c265;c266;c267;c268;c269;c270;c271;c272;c273;c274;c275;c276;c277;c278;c279;c280;c281;c282;c283;c284;c285;c286;c287;c288;c289;c290;c291;c292;c293;c294;c295;c296;c297;c298;c299;c300;c301;c302;c303;c304;c305;c306;c307;c308;c309;c310;c311;c312;c313;c314;c315;c316;c317;c318;c319;c320;c321;c322;c323;c324;c325;c326;c327;c328;c329;c330;c331;c332;c333;c334;c335;c336;c337;c338;c339;c340;c341;c342;c343;c344;c345;c346;c347;c348;c349;c350;c351;c352;c353;c354;c355;c356;c357;c358;c359;c360;c361;c362;c363;c364;c365;c366;c367;c368;c369;c370;c371;c372;c373;c374;c375;c376;c377;c378;c379;c380;c381;c382;c383;c384;c385;c386;c387;c388;c389;c390;c391;c392;c393;c394;c395;c396;c397;c398;c399;c400;classe\n')
    for arq in arqTreinos:
        arq=arq.replace('\n','')
        arq = arq.replace('/',separator)

        if(os.path.isfile('NCharacter_SD19_BMP'+arq)):
            novoCaminho = resize('NCharacter_SD19_BMP'+arq, arq)
            aux =''
         #   print(str(len(extrairCaracteristicas(novoCaminho))))
            for i in extrairCaracteristicas(novoCaminho):
                aux = aux+str(i) +';'
            linha = aux + arq[1:2]+'\n'
            treinoFinal.write(linha)
            """
    
       """
        cont = cont +1
        """
        if cont==5:
            break
            """
    treinoFinal.close()

main()