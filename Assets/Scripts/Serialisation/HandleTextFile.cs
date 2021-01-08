using UnityEngine;
using UnityEditor;
using System.IO;

public class HandleTextFile
{
    public static void WriteString(string path, string data)
    {
        //Write some text to the data.txt file
        StreamWriter writer = new StreamWriter(path, true);
        writer.WriteLine(data);
        writer.Close();
    }

    public static string ReadString(string path)
    {
        //Read the text from directly from the data.txt file
        StreamReader reader = new StreamReader(path); 
		string tmp = reader.ReadToEnd();
        reader.Close();
        return (tmp);
    }

}