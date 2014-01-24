package lda;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

public class extraksi_fitur {
    static double fiturlemon[][]=new double[3][4];
    static double fiturmanis[][]=new double[5][4];
    static double fiturnipis[][]=new double[7][4];
    static int cl=0,cm=0,cn=0;
public static void inputGambar(String namaFile){
        try {
            double redf=0,greenf=0,bluef=0;
            BufferedImage img = ImageIO.read(new File("CitraJeruk/Data Training/"+namaFile));
            BufferedImage img2 = new BufferedImage(img.getWidth(),img.getHeight(),BufferedImage.TYPE_INT_RGB);
            BufferedImage img3 = new BufferedImage(img.getWidth(),img.getHeight(),BufferedImage.TYPE_INT_RGB);
            int[] coordinat1=new int[img.getHeight()];
            int[] coordinat2=new int[img.getHeight()];
            for (int i=0; i<img.getHeight();i++){
                for (int j=0; j<img.getWidth();j++){
                    Color color = new Color(img.getRGB(j, i),true);
                    int red = color.getRed();
                    int green = color.getGreen();
                    int blue = color.getBlue();
                    if(red!=255 && blue!=255 & green!=255){
                    redf+=red;greenf+=green;bluef+=blue;}
                    int rata2 = Math.round((red+green+blue)/3);
                    int black = 0;
                    if(rata2>230) black=0;
                    else black=255;
                    black = (black<<16)|(black<<8)| black;
                    img3.setRGB(j, i, black);
                    rata2 = (rata2<<16)|(rata2<<8)| rata2;
                    img2.setRGB(j, i, rata2);                          
                }
            }
            int change=0,count1=0,count2=0;
            for(int i=0;i<img.getHeight();i++){
                for(int j=0;j<img.getWidth();j++){
                    Color c=new Color(img3.getRGB(j, i),true);
                    if(c.getRed()==255 && change==0){
                        coordinat1[count1++]=j;
                        change=1;
                    }
                    else if(c.getRed()==0 && change==1){
                        coordinat2[count2++]=j;
                        change=0;
                    }
                }
            }
            int min=99999,max=0;
            for(int i=0;i<count1;i++)
                if(min>coordinat1[i]) min=coordinat1[i];
            for(int i=0;i<count2;i++)
                if(max<coordinat2[i]) max=coordinat2[i];

            File abu2 = new File("abu2.jpg");
            ImageIO.write(img2, "jpg", abu2);
            File blackwhite = new File("blackwhite.jpg");
            ImageIO.write(img3, "jpg", blackwhite);
            if(namaFile.substring(5, 10).equalsIgnoreCase("Lemon")){
            fiturlemon[cl][0]=redf/(img.getWidth()*img.getHeight());
            fiturlemon[cl][1]=greenf/(img.getWidth()*img.getHeight());
            fiturlemon[cl][2]=bluef/(img.getWidth()*img.getHeight());
            fiturlemon[cl++][3]=max-min;
            }
            else if(namaFile.substring(5, 10).equalsIgnoreCase("Manis"))
            {
            fiturmanis[cm][0]=redf/(img.getWidth()*img.getHeight());
            fiturmanis[cm][1]=greenf/(img.getWidth()*img.getHeight());
            fiturmanis[cm][2]=bluef/(img.getWidth()*img.getHeight());
            fiturmanis[cm++][3]=max-min;
            }
            else if(namaFile.substring(5, 10).equalsIgnoreCase("Nipis"))
            {
            fiturnipis[cn][0]=redf/(img.getWidth()*img.getHeight());
            fiturnipis[cn][1]=greenf/(img.getWidth()*img.getHeight());
            fiturnipis[cn][2]=bluef/(img.getWidth()*img.getHeight());
            fiturnipis[cn++][3]=max-min;
            }
            System.out.println("Fitur RGB : "+redf/(img.getWidth()*img.getHeight())+" "+greenf/(img.getWidth()*img.getHeight())+" "+bluef/(img.getWidth()*img.getHeight()));
            System.out.println("Diameter : "+(max-min)+" pixel");
        } catch (IOException ex) {
            Logger.getLogger(extraksi_fitur.class.getName()).log(Level.SEVERE, null, ex);
        }
}
}
